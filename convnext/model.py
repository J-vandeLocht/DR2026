import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dino.utils import freeze_module
from transformer.adapters import build_adapter
from transformer.model import BagTransformer


CONVNEXT_WEIGHTS = {
    "large": "dino/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    "base": "dino/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
}


class ConvNextSelfAttention(BagTransformer):
    """DINOv3 ConvNeXt-L (or -B, backbone_arch="base") backbone -> adapter (feature map -> one or more
    tokens per frame) -> the same BagTransformer head as DinoSelfAttention, plus a final
    LayerNorm. No temporal encoding.

    backbone_weights="pretrained" loads the DINOv3 LVD weights for backbone_arch, None builds the backbone with random init (smoke tests);
    full_checkpoint_path loads a full ConvNextSelfAttention state dict
    strictly (architecture must match; prefer from_checkpoint(run_dir))."""

    def __init__(
            self,
            freeze_backbone=False,
            grad_checkpointing=True,
            adapter="linear_attnpool",
            adapter_dim=1024,
            adapter_pe=True,
            adapter_tokens=None,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=1,
            num_frames=32,
            num_classes=1,
            aggregation_mode="cls",
            num_query_tokens=1,
            feature_stage=4,
            feature_block=-1,
            backbone_arch="large",
            backbone_weights="pretrained",
            full_checkpoint_path=None,
    ):
        super().__init__()

        self.adapter_name = adapter
        self.grad_checkpointing = grad_checkpointing and not freeze_backbone

        self.backbone_arch = backbone_arch
        hub_name = f"dinov3_convnext_{backbone_arch}"
        if backbone_weights is not None and full_checkpoint_path is None:
            weights = CONVNEXT_WEIGHTS[backbone_arch] if backbone_weights == "pretrained" else backbone_weights
            self.backbone = torch.hub.load("dino/dinov3", hub_name, source="local", weights=weights)
        else:
            # Random init; full_checkpoint_path overwrites it below anyway.
            self.backbone = torch.hub.load("dino/dinov3", hub_name, source="local", pretrained=False)

        # Feature map taken after block `feature_block` (-1 = last) of stage `feature_stage` (1-4). The
        # pretrained final norm is only applied to the full backbone (DINOv3 itself uses no norm on
        # intermediate stages); later blocks/stages are frozen since they don't contribute.
        self.feature_stage = feature_stage
        self.feature_block = feature_block % len(self.backbone.stages[feature_stage - 1])
        self.use_final_norm = feature_stage == 4 and self.feature_block == len(self.backbone.stages[3]) - 1
        unused = [blk for i, stage in enumerate(self.backbone.stages) for j, blk in enumerate(stage)
                  if (i + 1, j) > (feature_stage, self.feature_block)]
        if not self.use_final_norm:
            unused.append(self.backbone.norm)
        unused += [self.backbone.downsample_layers[i] for i in range(feature_stage, 4)]
        for module in unused:
            freeze_module(module)

        C = self.backbone.embed_dims[feature_stage - 1]
        self.adapter = build_adapter(adapter, C, adapter_dim, adapter_tokens, adapter_pe)
        # Final norm of the pre-norm stack (as in ViT), before the classifier reads the tokens.
        self._build_head(adapter_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes,
                         aggregation_mode, num_query_tokens, final_norm=True,
                         tokens_per_frame=self.adapter.tokens_per_frame)
        print(
            f"--- ConvNeXt-{backbone_arch} Self-Attention [adapter={self.adapter_name} ({self.adapter.tokens_per_frame} token(s)/frame), "
            f"features from stage {feature_stage} "
            f"block {self.feature_block} ({C} ch), {self.L} features, "
            f"{self.num_blocks} block(s), aggregation={self.aggregation_mode}] weights initialized ---"
        )

        if full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        if freeze_backbone:
            freeze_module(self.backbone)

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS
    # ---------------------------------------------------------
    def backbone_feature_map(self, x):
        """[N, 3, H, W] -> [N, C, H/s, W/s] feature map after the configured stage/block
        (s = 4, 8, 16, 32 for stages 1-4; final norm only for the full backbone),
        checkpointing each ConvNeXt block when fine-tuning so a 32-frame bag fits in 24GB."""
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        stages = zip(self.backbone.downsample_layers[:self.feature_stage], self.backbone.stages[:self.feature_stage])
        for i, (downsample, stage) in enumerate(stages):
            x = downsample(x)
            last = self.feature_block if i == self.feature_stage - 1 else len(stage) - 1
            for block in stage[:last + 1]:
                x = checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)
        if self.use_final_norm:
            x = self.backbone.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def extract_features(self, x):
        # [N, 3, H, W] -> [N, L]: one vector per frame for the transformer head.
        return self.adapter(self.backbone_feature_map(x))
