"""Minimal LoRA for the DINOv3 ViT: wrap Linear layers, merge back into a plain DINOv3 state dict."""
import math

import torch
import torch.nn as nn

LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALinear(nn.Module):
    """base(x) + B(A(x)) * alpha / r. Calls the wrapped layer's own forward, so DINOv3's LinearKMaskedBias (qkv with
    bias_mask) keeps working. B is zero-initialized: the wrapped model starts out identical to the pretrained one."""

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features  # read by DINOv3's attention
        self.scale = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale

    def delta_weight(self):
        return (self.lora_B.weight @ self.lora_A.weight) * self.scale


def inject_lora(vit, r=16, alpha=32, dropout=0.0, targets=LORA_TARGETS):
    """Replace the `targets` Linear layers of every block with LoRALinear (in place); returns the LoRA modules."""
    wrapped = []
    for blk in vit.blocks:
        for name in targets:
            parent_name, attr = name.rsplit(".", 1)
            parent = blk.get_submodule(parent_name)
            lora = LoRALinear(getattr(parent, attr), r, alpha, dropout)
            setattr(parent, attr, lora)
            wrapped.append(lora)
    return wrapped


@torch.no_grad()
def merge_lora_state_dict(vit):
    """State dict of `vit` with every LoRALinear folded into its base weight, under the original (pre-LoRA) key names, on CPU,
    i.e. loadable with strict=True into a plain DINOv3 backbone (and usable as --backbone_weights)."""
    lora_prefixes = {name: m for name, m in vit.named_modules() if isinstance(m, LoRALinear)}
    out = {}
    for k, v in vit.state_dict().items():
        prefix = next((p for p in lora_prefixes if k.startswith(p + ".")), None)
        if prefix is None:
            out[k] = v.detach().cpu().clone()
            continue
        rest = k[len(prefix) + 1:]
        if rest.startswith(("lora_A.", "lora_B.")):
            continue
        assert rest.startswith("base."), k
        rest = rest[len("base."):]
        if rest == "weight":
            v = v + lora_prefixes[prefix].delta_weight().to(v.dtype)
        out[f"{prefix}.{rest}"] = v.detach().cpu().clone()
    return out
