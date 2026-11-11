"""Side-by-side comparison of the cropping models trained on Paxos2020 vs Paxos2025.

For each eval dataset (2020 / 2025) this picks --n random frames from the val splits
(the fold is chosen randomly per frame), runs both single-dataset models
(seg_2020_fold_<f>.pth and seg_2025_fold_<f>.pth of that fold) on the frame, and
writes one PNG per dataset. Each row has FOUR panels of equal size:

    RGB + ground-truth mask | RGB + 2020-model mask (IoU) | RGB + 2025-model mask (IoU)
    | difference map (green = both models, blue = only 2020, red = only 2025)

All panels of a row are cropped to ONE common square window at native resolution
(side = largest bbox dimension of the three masks, centered on their joint bbox,
clamped to the image) -- no resizing, no skewing -- so masks are pixel-comparable.
Predictions are mapped back to native frame resolution (nearest neighbor) first.

Outputs: <out_dir>/vis_compare_2020.png, <out_dir>/vis_compare_2025.png

Run from the repo root:
    python -u -m cropping.vis_compare_models
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import albumentations as albu
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from matplotlib.colors import ListedColormap

GT_COLOR = "lime"
PRED_COLORS = ["dodgerblue", "red", "darkorange", "purple"]


def get_val_transform(img_size):
    return albu.Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_gt_mask(row, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(row["center_x"]), int(row["center_y"])),
                (int(row["radius_x"]), int(row["radius_y"])), 0, 0, 360, 1, -1)
    return mask > 0


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def common_square_window(masks, h, w):
    """One square window shared by all panels: side = largest bbox dimension among
    the masks, centered on their joint bounding box, clamped to the image."""
    boxes = [b for b in (bbox(m) for m in masks) if b is not None]
    if not boxes:
        side = min(h, w)
        cy, cx = h // 2, w // 2
    else:
        side = min(max(max(b[1] - b[0], b[3] - b[2]) for b in boxes), h, w)
        cy = (min(b[0] for b in boxes) + max(b[1] for b in boxes)) // 2
        cx = (min(b[2] for b in boxes) + max(b[3] for b in boxes)) // 2
    r0 = min(max(cy - side // 2, 0), h - side)
    c0 = min(max(cx - side // 2, 0), w - side)
    return r0, c0, side


def iou(a, b, eps=1e-7):
    inter = (a & b).sum()
    return inter / (a.sum() + b.sum() - inter + eps)


def diff_panel(rgb, p1, p2):
    """RGB dimmed; mask pixels colored: green=both, blue=only p1, red=only p2."""
    out = rgb.astype(np.float32) * 0.35
    both = p1 & p2
    only1 = p1 & ~p2
    only2 = p2 & ~p1
    out[both] = (0.45, 1.0, 0.45)
    out[only1] = (0.35, 0.65, 1.0)
    out[only2] = (1.0, 0.35, 0.35)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_model(path, encoder_name, device):
    model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None,
                             in_channels=3, classes=1).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="cropping/models_xdomain")
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="cropping/eval_xdomain")
    parser.add_argument("--img_size", type=int, default=768, help="model input size (as trained)")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--n", type=int, default=15, help="frames per eval dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--configs", nargs="+", default=["2020", "2025"], help="the two models to compare")
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 9})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transform(args.img_size)
    rng = random.Random(args.seed)
    assert len(args.configs) == 2, "exactly two models are compared (--configs)"
    c1, c2 = args.configs

    for eval_ds in args.eval_datasets:
        df = pd.read_csv(Path(args.ann_dir) / f"annotations_cropping_{eval_ds}.csv").set_index("filename")
        with open(Path(args.ann_dir) / f"splits_{eval_ds}.json") as f:
            splits = json.load(f)
        candidates = [(fold, fn) for fold in range(args.n_splits) for fn in splits[f"fold_{fold}"]["val"]]
        picks = rng.sample(candidates, min(args.n, len(candidates)))

        models = {}
        for fold, _ in picks:
            for cfg in args.configs:
                if (cfg, fold) not in models:
                    path = Path(args.models_dir) / f"seg_{cfg}_fold_{fold}.pth"
                    if not path.exists():
                        print(f"WARNING: missing {path}")
                        continue
                    models[(cfg, fold)] = load_model(path, args.encoder_name, device)

        rows = []
        for fold, fn in picks:
            if (c1, fold) not in models or (c2, fold) not in models:
                continue
            img_bgr = cv2.imread(str(Path(args.img_dir) / fn))
            if img_bgr is None:
                print(f"WARNING: cannot read {args.img_dir}/{fn} -- skipped")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            gt = build_gt_mask(df.loc[fn], h, w)
            inp = transform(image=img)["image"].unsqueeze(0).to(device)
            preds = {}
            with torch.no_grad():
                for cfg in (c1, c2):
                    p = (torch.sigmoid(models[(cfg, fold)](inp))[0, 0] > 0.5).cpu().numpy()
                    preds[cfg] = cv2.resize(p.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            r0, c0, side = common_square_window([gt, preds[c1], preds[c2]], h, w)
            rows.append({
                "fn": fn, "fold": fold, "side": side,
                "rgb": img[r0:r0 + side, c0:c0 + side],
                "gt": gt[r0:r0 + side, c0:c0 + side],
                "p1": preds[c1][r0:r0 + side, c0:c0 + side],
                "p2": preds[c2][r0:r0 + side, c0:c0 + side],
            })

        if not rows:
            print(f"eval_ds={eval_ds}: nothing to plot (missing models/images)")
            continue

        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(5.0 * 4, 4.6 * n))
        if n == 1:
            axes = axes[None, :]
        for j, t in enumerate(["ground truth", f"model {c1}", f"model {c2}", f"diff {c1} vs {c2}"]):
            axes[0, j].set_title(t, fontsize=12, pad=8)
        for i, r in enumerate(rows):
            axes[i, 0].set_ylabel("\n".join(textwrap.wrap(r["fn"], 32)) + f"\nfold {r['fold']} | {r['side']}px",
                                  fontsize=8)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[i, 0].imshow(r["rgb"], interpolation="nearest")
            axes[i, 0].imshow(np.ma.masked_where(~r["gt"], r["gt"]),
                              cmap=ListedColormap(["none", GT_COLOR]), alpha=0.55, interpolation="nearest")
            axes[i, 1].imshow(r["rgb"], interpolation="nearest")
            axes[i, 1].imshow(np.ma.masked_where(~r["p1"], r["p1"]),
                              cmap=ListedColormap(["none", PRED_COLORS[0]]), alpha=0.55, interpolation="nearest")
            axes[i, 1].set_title(f"IoU {iou(r['p1'], r['gt']):.3f}", fontsize=11)
            axes[i, 2].imshow(r["rgb"], interpolation="nearest")
            axes[i, 2].imshow(np.ma.masked_where(~r["p2"], r["p2"]),
                              cmap=ListedColormap(["none", PRED_COLORS[1]]), alpha=0.55, interpolation="nearest")
            axes[i, 2].set_title(f"IoU {iou(r['p2'], r['gt']):.3f}", fontsize=11)
            axes[i, 3].imshow(diff_panel(r["rgb"], r["p1"], r["p2"]), interpolation="nearest")
            axes[i, 3].set_title("green=both, blue=2020 only, red=2025 only", fontsize=9)
        fig.suptitle(f"Cropping on Paxos{eval_ds} val frames: model {c1} vs model {c2} "
                     f"(random fold per frame; RGB under mask; one common square window per row, no resize)",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"vis_compare_{eval_ds}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {out_path} ({n} frames)")


if __name__ == "__main__":
    main()
