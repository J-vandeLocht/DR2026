"""Self-supervised LoRA domain adaptation of DINOv3 ViT-L/16 on Paxos2025 frames (ExPLoRA-style): DINOv3's own objective
(DINO CLS + iBOT masked patch, Sinkhorn-Knopp teacher targets, KoLeo, Gram anchoring to the frozen starting backbone)
with an EMA teacher; only LoRA adapters, the final norm and two fresh DINO heads train. Frames come from data/extract_ssl_frames.py (Paxos2025 and/or Paxos2020, border-cropped, CLAHE applied here).
Writes <out_dir>/<run>/merged_backbone.pth (teacher, LoRA merged, plain DINOv3 keys -> --backbone_weights of
dino/train_dino_classifier.py). Single GPU: python -m dino.ssl_lora; multi-GPU: torchrun --nproc_per_node N -m dino.ssl_lora."""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from data.dataset import crop_black_borders
from dino.lora import LORA_TARGETS, LoRALinear, inject_lora, merge_lora_state_dict
from dino.utils import DINO_REPO, DINO_WEIGHTS, clahe_rgb, load_dino_backbone, make_val_transform_dino
from misc.training import loader_kwargs, seed_everything

sys.path.insert(0, DINO_REPO)
from dinov3.data.masking import MaskingGenerator  # noqa: E402
from dinov3.layers.dino_head import DINOHead  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
class MultiCropFrames(Dataset):
    """All extracted frames; each item -> (list of 2 global + n_local crops, list of 0 or 2 Gram-teacher views), all
    normalized tensors. A Gram-teacher view is its global crop's region (same rotation/crop/flips) at gram_size,
    without color jitter or blur (DINOv3's gram_teacher_no_distortions)."""

    def __init__(self, frame_dir, global_size, local_size, n_local, global_scale, local_scale, clahe=True,
                 frame_list=None, blur_p=(0.5, 0.1, 0.2), blur_sigma=1.0, gram_size=None):
        """blur_p: Gaussian-blur probability of (global view 1, global view 2, local crops); sigma in [0.1, blur_sigma].
        gram_size: also return the Gram-teacher views (None: don't)."""
        if frame_list:  # paths relative to frame_dir, e.g. data/ssl_frame_quality.py's keep_p10.txt
            with open(frame_list) as f:
                self.paths = sorted(os.path.join(frame_dir, line.strip()) for line in f if line.strip())
        else:
            self.paths = sorted(glob.glob(os.path.join(frame_dir, "*", "*.jpg")))
        if not self.paths:
            raise FileNotFoundError(f"No frames in {frame_dir}; run python -m data.extract_ssl_frames first")
        self.clahe, self.n_local = clahe, n_local
        self.global_size, self.global_scale, self.gram_size = global_size, global_scale, gram_size
        color = v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)], p=0.8)
        geo = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
        self.norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
        blur = lambda p: v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, blur_sigma))], p=p)  # noqa: E731
        self.rot = v2.RandomRotation(180)  # before the crop, so crops don't get extra black corners
        # globals: geometry in global_views (shared with the Gram-teacher view), then these photometric augs
        self.global_photo = [v2.Compose([color, blur(blur_p[0]), self.norm]), v2.Compose([color, blur(blur_p[1]), self.norm])]
        self.local = v2.Compose([self.rot, v2.RandomResizedCrop(local_size, scale=local_scale, antialias=True),
                                 *geo, color, blur(blur_p[2]), self.norm])

    def global_views(self, img, k):
        """Global crop k (+ its undistorted Gram-teacher view, or None): rotation, random resized crop, flips."""
        img = self.rot(img)
        i, j, h, w = v2.RandomResizedCrop.get_params(img, self.global_scale, (3 / 4, 4 / 3))
        crop = img[:, i:i + h, j:j + w]
        if random.random() < 0.5:
            crop = crop.flip(-1)
        if random.random() < 0.5:
            crop = crop.flip(-2)
        g = self.global_photo[k](v2.functional.resize(crop, [self.global_size] * 2, antialias=True))
        t = self.norm(v2.functional.resize(crop, [self.gram_size] * 2, antialias=True)) if self.gram_size else None
        return g, t

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        if self.clahe:
            img = clahe_rgb(img)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # uint8 CHW
        (g1, t1), (g2, t2) = self.global_views(img, 0), self.global_views(img, 1)
        return [g1, g2] + [self.local(img) for _ in range(self.n_local)], ([t1, t2] if self.gram_size else [])


class MaskCollate:
    """Stack crops (+ Gram-teacher views, else None) and draw iBOT block masks for a `mask_prob` share of the global
    crops, ratios spread over mask_ratio (as in DINOv2/v3's collate)."""

    def __init__(self, n_tokens_side, mask_ratio, mask_prob):
        self.n_tokens = n_tokens_side ** 2
        self.gen = MaskingGenerator((n_tokens_side, n_tokens_side), max_num_patches=int(0.5 * self.n_tokens))
        self.mask_ratio, self.mask_prob = mask_ratio, mask_prob

    def __call__(self, batch):
        grams = torch.stack([s[1][c] for c in range(2) for s in batch]) if batch[0][1] else None  # [2B, ...]
        batch = [s[0] for s in batch]
        n_crops = len(batch[0])
        globals_ = torch.stack([s[c] for c in range(2) for s in batch])  # [2B, ...], crop-major
        locals_ = torch.stack([s[c] for c in range(2, n_crops) for s in batch]) if n_crops > 2 else None
        n = len(globals_)
        n_masked = int(n * self.mask_prob)
        probs = np.linspace(*self.mask_ratio, n_masked + 1)
        masks = [torch.from_numpy(self.gen(int(self.n_tokens * random.uniform(probs[i], probs[i + 1]))))
                 for i in range(n_masked)]
        masks += [torch.zeros(int(self.n_tokens ** 0.5), int(self.n_tokens ** 0.5), dtype=torch.bool)] * (n - n_masked)
        random.shuffle(masks)
        return globals_, locals_, torch.stack(masks).flatten(1), grams


# ---------------------------------------------------------------- model
class SSLModel(nn.Module):
    def __init__(self, backbone, n_prototypes, hidden_dim, bottleneck_dim):
        super().__init__()
        self.backbone = backbone
        d = backbone.embed_dim
        self.dino_head = DINOHead(d, n_prototypes, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.ibot_head = DINOHead(d, n_prototypes, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.dino_head.init_weights()
        self.ibot_head.init_weights()

    def forward(self, globals_, locals_, masks, mask_input):
        """-> global CLS feats [2B, D], DINO logits of global [2B, K] and local [nB, K] crops (None without locals),
        iBOT logits of the masked global patches [M, K], normed global patch tokens [2B, P, D] (Gram loss).
        mask_input: replace masked patches by the mask token (student); the teacher sees the full crops and `masks`
        only selects its target patches."""
        in_masks = masks if mask_input else None
        if locals_ is None:
            outs = [self.backbone.forward_features(globals_, in_masks)]
        else:
            outs = self.backbone.forward_features([globals_, locals_], [in_masks, None])
        g_cls = outs[0]["x_norm_clstoken"]
        g_logits = self.dino_head(g_cls)
        l_logits = self.dino_head(outs[1]["x_norm_clstoken"]) if locals_ is not None else None
        patches = outs[0]["x_norm_patchtokens"]
        p_logits = self.ibot_head(patches[masks])
        return g_cls, g_logits, l_logits, p_logits, patches


# ---------------------------------------------------------------- losses (single-GPU safe)
def _all_reduce(t):
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


@torch.no_grad()
def sinkhorn_knopp(logits, temp, n_iter=3):
    """Teacher targets [N, K] (rows sum to 1), balanced over the prototypes across all ranks."""
    Q = torch.exp((logits.float() - logits.float().max()) / temp).t()  # [K, N]; global max shift cancels out
    B = _all_reduce(torch.tensor(Q.shape[1], device=Q.device, dtype=torch.float32))
    K = Q.shape[0]
    Q /= _all_reduce(Q.sum())
    for _ in range(n_iter):
        Q /= _all_reduce(Q.sum(dim=1, keepdim=True))
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


def dino_loss(student_logits, teacher_probs, temp, local_weight=1.0):
    """student [S, B, K] (first 2 = global crops), teacher [2, B, K]; CE over all pairs except same view. As in DINOv3:
    global and local pair means weighted by their pair counts, the local one also by local_weight (1: plain mean over
    all pairs). -> (loss, global pair mean, local pair mean)."""
    S, B, _ = student_logits.shape
    log_p = F.log_softmax(student_logits.float() / temp, dim=-1)
    loss = -torch.einsum("sbk,tbk->st", log_p, teacher_probs) / B  # [S, 2] per-pair means
    loss = torch.diagonal_scatter(loss, loss.new_zeros(2))
    n_glob, n_loc = 2, 2 * (S - 2)
    glob = loss[:2].sum() / n_glob
    loc = loss[2:].sum() / max(n_loc, 1)
    return (n_glob * glob + local_weight * n_loc * loc) / (n_glob + n_loc), glob, loc


def gram_loss(student_patches, teacher_patches):
    """DINOv3 Gram anchoring (img_level, normalized, no negative removal): MSE between the per-image patch cosine-
    similarity matrices of student [N, P, D] and Gram teacher [N, P, D]."""
    s = F.normalize(student_patches.float(), dim=-1)
    t = F.normalize(teacher_patches.float(), dim=-1)
    return F.mse_loss(s @ s.transpose(1, 2), t @ t.transpose(1, 2))


@torch.no_grad()
def gram_teacher_patches(backbone, imgs, out_side, chunk=16):
    """Normed patch tokens of the frozen Gram teacher [N, out_side^2, D], resized bicubically to the student's patch
    grid when imgs have another size (DINOv3 feeds the Gram teacher larger crops). Chunked to cap activation memory;
the weights must stay fp32 (fp16 weights give NaNs)."""
    out = []
    for x in imgs.split(chunk):
        p = backbone.forward_features(x)["x_norm_patchtokens"]
        side = int(p.shape[1] ** 0.5)
        if side != out_side:
            p = F.interpolate(p.float().transpose(1, 2).unflatten(-1, (side, side)), size=(out_side, out_side),
                              mode="bicubic", align_corners=False).flatten(2).transpose(1, 2)
        out.append(p)
    return torch.cat(out)


def ibot_loss(student_logits, teacher_probs, masks, temp):
    """Masked-patch CE, each image weighted by 1 / its number of masked patches, averaged over all global crops."""
    w = (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]
    ce = torch.sum(teacher_probs * F.log_softmax(student_logits.float() / temp, dim=-1), dim=-1)
    return -(ce * w).sum() / masks.shape[0]


def koleo_loss(x, eps=1e-8):
    x = F.normalize(x.float(), dim=-1, eps=eps)
    dots = x @ x.t()
    dots.view(-1)[:: len(x) + 1].fill_(-1)
    nn_idx = dots.argmax(dim=1)
    return -torch.log(F.pairwise_distance(x, x[nn_idx], eps=eps) + eps).mean()


# ---------------------------------------------------------------- schedules
def cosine(step, total, start, end):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step, total) / total))


def frac_steps(frac, args):
    return int(round(frac * args.steps))


def lr_at(step, args):
    """Heads (+ everything outside the backbone): linear warm-up, then cosine to min_lr."""
    warmup = max(frac_steps(args.warmup_frac, args), 1)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return cosine(step - warmup, args.steps - warmup, args.lr, args.min_lr)


def backbone_lr_at(step, args):
    """LoRA (+ final norm): 0 during the head warm-up (heads settle on fixed features while the teacher temperature
    warms up), then a linear ramp to lr * backbone_lr_mult and its own cosine to min_lr over the remaining steps."""
    start = frac_steps(args.head_warmup_frac, args)
    if step < start:
        return 0.0
    peak = args.lr * args.backbone_lr_mult
    ramp = max(frac_steps(args.backbone_ramp_frac, args), 1)
    if step - start < ramp:
        return peak * (step - start + 1) / ramp
    return cosine(step - start - ramp, args.steps - start - ramp, peak, args.min_lr)


# ---------------------------------------------------------------- probe
class StillsProbe:
    """Logistic-regression probe (5-class, QWK) on frozen teacher CLS features of the labeled stills, per split
    (frame_train -> frame_val), averaged over all splits. Images are border-cropped + CLAHE'd as in training."""

    def __init__(self, split_dir, frame_dir, img_size, device):
        self.splits = sorted(d for d in glob.glob(os.path.join(split_dir, "split_*")) if os.path.isdir(d))
        labels = {}
        for s in self.splits:
            for f in ("frame_train.json", "frame_val.json"):
                labels.update(json.load(open(os.path.join(s, f))))
        self.names = sorted(labels)
        self.grades = np.array([labels[n] for n in self.names])
        self.frame_dir, self.device = frame_dir, device
        self.tf = make_val_transform_dino(img_size, complex_augs=True)

    def load(self, names):
        return torch.stack([self.tf(Image.fromarray(crop_black_borders(np.asarray(
            Image.open(os.path.join(self.frame_dir, n)).convert("RGB"))))) for n in names]).to(self.device)

    @torch.no_grad()
    def features(self, backbone, bs=16):
        was_training = backbone.training
        backbone.eval()
        feats = []
        for i in range(0, len(self.names), bs):
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(backbone.forward_features(self.load(self.names[i:i + bs]))["x_norm_clstoken"].float().cpu())
        backbone.train(was_training)
        return torch.cat(feats).numpy()

    @torch.no_grad()
    def patch_drift(self, backbone, ref_backbone, n=64, bs=16):
        """Gram loss between the patch tokens of `backbone` and `ref_backbone` (the pretrained one) on the first n
        stills (fixed subset): how far the dense similarity structure has drifted (0 at the start)."""
        was_training = backbone.training
        backbone.eval()
        losses = []
        for i in range(0, min(n, len(self.names)), bs):
            x = self.load(self.names[i:i + bs])
            with torch.autocast("cuda", dtype=torch.float16):
                losses.append(gram_loss(backbone.forward_features(x)["x_norm_patchtokens"],
                                        ref_backbone.forward_features(x)["x_norm_patchtokens"]).item())
        backbone.train(was_training)
        return float(np.mean(losses))

    def __call__(self, backbone):
        X = self.features(backbone)
        idx = {n: i for i, n in enumerate(self.names)}
        qwks = []
        for s in self.splits:
            tr = [idx[n] for n in json.load(open(os.path.join(s, "frame_train.json")))]
            va = [idx[n] for n in json.load(open(os.path.join(s, "frame_val.json")))]
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr]), self.grades[tr])
            qwks.append(cohen_kappa_score(self.grades[va], clf.predict(sc.transform(X[va])), weights="quadratic"))
        return float(np.mean(qwks)), qwks


# ---------------------------------------------------------------- main
def trainable_named(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def main(args):
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda")
    main_proc = rank == 0
    seed_everything(args.seed + rank)

    # Student: pretrained backbone + LoRA (+ optional last blocks) + final norm + fresh heads.
    backbone = load_dino_backbone(args.backbone_weights)
    for p in backbone.parameters():
        p.requires_grad = False
    inject_lora(backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, targets=args.lora_targets)
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.weight.requires_grad = m.lora_B.weight.requires_grad = True
    for p in backbone.norm.parameters():
        p.requires_grad = True
    for blk in backbone.blocks[len(backbone.blocks) - args.unfreeze_last_blocks:] if args.unfreeze_last_blocks else []:
        for n, p in blk.named_parameters():
            p.requires_grad = True
    backbone.grad_checkpointing = not args.no_grad_checkpointing
    student = SSLModel(backbone, args.n_prototypes, args.head_hidden_dim, args.head_bottleneck_dim).to(device)
    # Gram teacher (DINOv3 Gram anchoring): the frozen starting backbone, also the reference of the probe's patch drift
    gram_teacher = None
    if args.gram_weight > 0 or (args.probe and main_proc):
        gram_teacher = load_dino_backbone(args.backbone_weights).to(device).eval()
        gram_teacher.requires_grad_(False)

    params = trainable_named(student)
    groups = {}  # (is_backbone, decay) -> params; backbone groups get lr 0 during the head warm-up
    for n, p in params.items():
        groups.setdefault((n.startswith("backbone."), p.ndim >= 2 and "norm" not in n), []).append(p)
    optimizer = torch.optim.AdamW([{"params": ps, "weight_decay": args.weight_decay if decay else 0.0,
                                    "is_backbone": is_bb} for (is_bb, decay), ps in groups.items()],
                                  lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    n_backbone = sum(p.numel() for n, p in params.items() if n.startswith("backbone."))
    n_head = sum(p.numel() for n, p in params.items() if not n.startswith("backbone."))

    run_dir, start_step = None, 0
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume, "ckpt.pth"), map_location="cpu")
        student.load_state_dict(ckpt["student"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step, run_dir = ckpt["step"], args.resume
    run_name = os.path.basename(run_dir) if run_dir else (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_ssl_lora_r{args.lora_r}{f'_{args.run_tag}' if args.run_tag else ''}")
    run_dir = run_dir or os.path.join(args.out_dir, run_name)

    # DDP broadcasts rank 0's student (random head/LoRA init differs per rank), so the teacher is copied after it.
    model = DDP(student, device_ids=[device.index]) if distributed else student
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if args.resume:
        teacher.load_state_dict(ckpt["teacher"], strict=False)

    dataset = MultiCropFrames(args.frame_dir, args.global_size, args.local_size, args.n_local,
                              tuple(args.global_scale), tuple(args.local_scale), clahe=not args.no_clahe,
                              frame_list=args.frame_list, blur_p=tuple(args.blur_p), blur_sigma=args.blur_sigma,
                              gram_size=args.gram_teacher_size if args.gram_weight > 0 else None)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True) \
        if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True,
                        collate_fn=MaskCollate(args.global_size // 16, tuple(args.mask_ratio), args.mask_prob),
                        pin_memory=True, **loader_kwargs(args.num_workers))

    writer, probe = None, None
    if main_proc:
        os.makedirs(run_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=2)
        writer = SummaryWriter(os.path.join("runs", run_name))
        print(f"Run {run_name}: {len(dataset)} frames, {world} GPU(s) x bs {args.batch_size}, "
              f"trainable backbone {n_backbone / 1e6:.2f}M + heads {n_head / 1e6:.2f}M")
        if args.probe:
            probe = StillsProbe(args.probe_split_dir, args.probe_frame_dir, args.probe_img_size, device)

    def run_probe(step):
        if probe is not None:
            mean_qwk, qwks = probe(teacher.backbone)
            drift = probe.patch_drift(teacher.backbone, gram_teacher)
            writer.add_scalar("probe/qwk_mean", mean_qwk, step)
            writer.add_scalar("probe/patch_gram_drift", drift, step)
            print(f"[probe step {step}] stills LR-probe QWK {mean_qwk:.3f} ({' '.join(f'{q:.3f}' for q in qwks)}), "
                  f"patch Gram drift vs pretrained {drift:.2e}", flush=True)
            torch.cuda.empty_cache()  # probe activations at 512 px would otherwise stay cached next to training
        if distributed:
            dist.barrier()

    def save(step, keep=False):
        if distributed:  # the EMA teachers must stay identical across ranks
            chk = torch.stack([p.double().sum() for p in teacher_params.values()]).sum().reshape(1)
            gathered = [torch.zeros_like(chk) for _ in range(world)]
            dist.all_gather(gathered, chk)
            if main_proc:
                print(f"[step {step}] teacher checksum per rank: {[f'{g.item():.6f}' for g in gathered]}", flush=True)
        if main_proc:
            tstate = {n: p for n, p in teacher.state_dict().items()
                      if n in params or not n.startswith("backbone.")}  # trainable + heads; base comes from the weights
            torch.save({"student": {n: p.detach().cpu() for n, p in params.items()},
                        "teacher": tstate, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "args": vars(args)}, os.path.join(run_dir, "ckpt.pth"))
            merged = merge_lora_state_dict(teacher.backbone)
            torch.save(merged, os.path.join(run_dir, "merged_backbone.pth"))
            if keep:  # one per probe step, to pick the best probe step afterwards
                torch.save(merged, os.path.join(run_dir, f"merged_backbone_step{step}.pth"))
            print(f"[step {step}] saved {run_dir}/{{ckpt,merged_backbone}}.pth", flush=True)
        if distributed:
            dist.barrier()

    eval_every = max(args.steps // args.n_evals, 1)
    if start_step == 0 and args.probe:
        run_probe(0)

    teacher_params = {n: p for n, p in teacher.named_parameters() if n in params}
    step, epoch, t0 = start_step, 0, time.time()
    model.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for globals_, locals_, masks, grams in loader:
            if step >= args.steps:
                break
            lr, bb_lr = lr_at(step, args), backbone_lr_at(step, args)
            for g in optimizer.param_groups:
                g["lr"] = bb_lr if g["is_backbone"] else lr
            t_warm = max(frac_steps(args.teacher_temp_warmup_frac, args), 1)
            t_temp = (args.teacher_temp_start + (args.teacher_temp - args.teacher_temp_start) * step
                      / t_warm) if step < t_warm else args.teacher_temp
            momentum = cosine(step, args.steps, args.momentum_start, 1.0)
            globals_, locals_, masks = (globals_.to(device, non_blocking=True), locals_.to(device, non_blocking=True),
                                        masks.to(device, non_blocking=True))
            B = globals_.shape[0] // 2

            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    _, t_logits, _, t_plogits, _ = teacher(globals_, None, masks, mask_input=False)
                    t_probs = sinkhorn_knopp(t_logits, t_temp).view(2, B, -1)
                    t_pprobs = sinkhorn_knopp(t_plogits, t_temp)
                    g_patches = gram_teacher_patches(gram_teacher, grams.to(device, non_blocking=True),
                                                     args.global_size // 16) if grams is not None else None
                g_cls, g_logits, l_logits, p_logits, s_patches = model(globals_, locals_, masks, mask_input=True)
                s_logits = torch.cat([g_logits.view(2, B, -1), l_logits.view(args.n_local, B, -1)])
                l_dino, l_dino_g, l_dino_l = dino_loss(s_logits, t_probs, args.student_temp, args.local_loss_weight)
                l_ibot = ibot_loss(p_logits, t_pprobs, masks, args.student_temp)
                l_koleo = sum(koleo_loss(c) for c in g_cls.view(2, B, -1))
                l_gram = gram_loss(s_patches, g_patches) if g_patches is not None else torch.zeros((), device=device)
                loss = (args.dino_weight * l_dino + args.ibot_weight * l_ibot + args.koleo_weight * l_koleo
                        + args.gram_weight * l_gram)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if step < frac_steps(args.freeze_last_layer_frac, args):
                for head in (student.dino_head, student.ibot_head):
                    for p in head.last_layer.parameters():
                        p.grad = None
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(list(params.values()), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():  # EMA teacher (identical on all ranks: DDP keeps student params in sync)
                names = list(teacher_params)
                torch._foreach_mul_([teacher_params[n] for n in names], momentum)
                torch._foreach_add_([teacher_params[n] for n in names], [params[n].detach() for n in names],
                                    alpha=1 - momentum)
            step += 1

            if main_proc and (step % args.log_every == 0 or step == 1):
                with torch.no_grad():
                    t_ent = -(t_probs * torch.log(t_probs + 1e-12)).sum(-1).mean().item()
                    s_ent = -(F.softmax(g_logits.float() / args.student_temp, -1)
                              * F.log_softmax(g_logits.float() / args.student_temp, -1)).sum(-1).mean().item()
                    feat_std = F.normalize(g_cls.float(), dim=-1).std(0).mean().item()
                dt = (time.time() - t0) / (step - start_step)
                for k, v in {"loss/total": loss.item(), "loss/dino": l_dino.item(), "loss/ibot": l_ibot.item(),
                             "loss/koleo": l_koleo.item(), "loss/gram": l_gram.item(),
                             "loss/dino_global": l_dino_g.item(), "loss/dino_local": l_dino_l.item(),
                             "train/lr": lr, "train/backbone_lr": bb_lr,
                             "train/teacher_temp": t_temp,
                             "train/momentum": momentum, "train/grad_norm": gnorm.item(),
                             "train/teacher_entropy": t_ent, "train/student_entropy": s_ent,
                             "train/cls_feat_std": feat_std, "train/sec_per_step": dt}.items():
                    writer.add_scalar(k, v, step)
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"step {step}/{args.steps} loss {loss.item():.3f} (dino {l_dino.item():.3f} ibot {l_ibot.item():.3f}"
                      f" koleo {l_koleo.item():.3f} gram {l_gram.item():.5f}) lr {lr:.2e}/{bb_lr:.2e} T {t_temp:.3f} m {momentum:.4f} |g| {gnorm.item():.2f}"
                      f" featstd {feat_std:.4f} {dt:.2f}s/step {mem:.1f}GB "
                      f"eta {(args.steps - step) * dt / 3600:.1f}h", flush=True)
            if step % eval_every == 0 or step == args.steps:
                if args.probe:
                    run_probe(step)
                save(step, keep=True)
        epoch += 1

    if main_proc:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame_dir", default="data/ensemble_results_paxos2025/ssl_frames")
    p.add_argument("--frame_list", default=None,
                   help="Only these frames (paths relative to --frame_dir), e.g. the keep_p10.txt of data/ssl_frame_quality.py.")
    p.add_argument("--out_dir", default="dino/ssl_lora")
    p.add_argument("--backbone_weights", default=DINO_WEIGHTS)
    p.add_argument("--resume", default=None, help="Run dir to resume from (ckpt.pth).")
    p.add_argument("--run_tag", default=None)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", nargs="+", default=list(LORA_TARGETS))
    p.add_argument("--unfreeze_last_blocks", type=int, default=0, help="Fully train the last k blocks (ExPLoRA: 1).")
    # heads / losses
    p.add_argument("--n_prototypes", type=int, default=16384)
    p.add_argument("--head_hidden_dim", type=int, default=2048)
    p.add_argument("--head_bottleneck_dim", type=int, default=256)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp_start", type=float, default=0.04)
    p.add_argument("--teacher_temp", type=float, default=0.07)
    p.add_argument("--teacher_temp_warmup_frac", type=float, default=0.3, help="Share of steps for the teacher-temp warm-up.")
    p.add_argument("--momentum_start", type=float, default=0.994)
    p.add_argument("--dino_weight", type=float, default=1.0)
    p.add_argument("--ibot_weight", type=float, default=1.0)
    p.add_argument("--koleo_weight", type=float, default=0.1)
    p.add_argument("--local_loss_weight", type=float, default=0.5,
                   help="Weight of the local-crop DINO pairs vs the global ones (DINOv3 gram/high-res stages: 0.5; v2: 1).")
    p.add_argument("--gram_weight", type=float, default=1.0,
                   help="DINOv3 Gram anchoring of the student's global patch tokens to the frozen starting backbone "
                        "(DINOv3: 1-2; 0 = off, v2).")
    p.add_argument("--gram_teacher_size", type=int, default=512,
                   help="Gram-teacher view size (DINOv3: 2x the global size; 512 = the frames' native resolution).")
    # crops / masks
    p.add_argument("--global_size", type=int, default=384, help="Global crop size (multiple of 16; v1: 256).")
    p.add_argument("--local_size", type=int, default=112)
    p.add_argument("--n_local", type=int, default=8)
    p.add_argument("--global_scale", type=float, nargs=2, default=[0.32, 1.0])
    p.add_argument("--local_scale", type=float, nargs=2, default=[0.025, 0.12],
                   help="Area share of the 512 px frame per local crop: 81-177 px -> 112 px, i.e. 0.63-1.38x magnification "
                        "around native, so small lesions survive (v1: 0.05 0.32, 0.39-0.97x).")
    p.add_argument("--blur_p", type=float, nargs=3, default=[0.5, 0.1, 0.2],
                   help="Blur probability of global view 1, global view 2, local crops (v1/DINO: 1.0 0.1 0.5).")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                   help="Max blur sigma in px; less blur invariance keeps microaneurysm-scale detail (v1/DINO: 2.0).")
    p.add_argument("--mask_ratio", type=float, nargs=2, default=[0.1, 0.5])
    p.add_argument("--mask_prob", type=float, default=0.5)
    p.add_argument("--no_clahe", action="store_true")
    # optimization
    p.add_argument("--steps", type=int, default=2550, help="4 GPUs x bs 64 x 2550 = 653k images (v1: 4 x 32 x 5000 = 640k); ~5 epochs of the "
                        "126,909 Paxos2025 + 2020 frames, ~10 of the 64,928 2025 ones.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Per GPU (384 px globals on a TITAN RTX, without Gram: bs 32 10.8 GB / 5.3 s, bs 48 14.7 GB / 8.2 s, "
                        "bs 64 18.7 GB / 10.6 s per step; with Gram (v3) bs 64 20.6 GB / ~13 s, needs expandable_segments).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    # Schedule lengths are shares of --steps, so they scale with it (v1, 5000 steps: warm-up 0.1, head warm-up 0.1 with
    # LoRA starting at the full lr, teacher temp 0.3, last layer 0.1, LoRA lr = head lr).
    p.add_argument("--backbone_lr_mult", type=float, default=0.25, help="LoRA (+ final norm) peak lr = lr * this.")
    p.add_argument("--warmup_frac", type=float, default=0.05, help="Linear lr warm-up of the heads.")
    p.add_argument("--head_warmup_frac", type=float, default=0.3,
                   help="Backbone lr is 0 for this share of steps (heads settle while the teacher temp warms up).")
    p.add_argument("--backbone_ramp_frac", type=float, default=0.1,
                   help="Linear backbone lr ramp after the head warm-up, then its own cosine to min_lr.")
    p.add_argument("--freeze_last_layer_frac", type=float, default=0.05, help="Heads' last layer frozen for this share.")
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--no_grad_checkpointing", action="store_true", help="Faster, more memory.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Per GPU (persistent; bs 64 at 384 px loads in ~0.5 s/batch (+ Gram views) vs a 10.5-13 s GPU step).")
    # logging / probe
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--n_evals", type=int, default=5,
                   help="Probe + checkpoint (incl. a kept merged_backbone_step<N>.pth) every steps / n_evals steps.")
    p.add_argument("--no_probe", dest="probe", action="store_false", help="Skip the stills probe.")
    p.add_argument("--probe_split_dir", default="data/stratified_splits_paxos2025_patient_q5_v2")
    p.add_argument("--probe_frame_dir", default="data/2024_Paxos_Frames/cropped_matched_frames")
    p.add_argument("--probe_img_size", type=int, default=512)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
