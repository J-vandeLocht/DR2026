"""Side-by-side mask comparison per val frame: in-dataset model vs cross-dataset ensemble.

For each eval dataset (2020 / 2025) this picks --n random frames from the val splits
(the fold is chosen randomly per frame) and writes one PNG per frame into
<out_dir>/vis_compare_<ds>/. Each figure has THREE panels of equal size:

    ground-truth region | same-dataset model of that fold (IoU) | ensemble (IoU)

- "same-dataset model": seg_<ds>_fold_<f>.pth for the frame's fold f -- the model
  that was NOT trained on the split the frame is in (val fold f).
- "ensemble": mean of sigmoid PROBABILITIES (threshold 0.5) over all 5 folds of the
  OTHER dataset's model (seg_<other>_fold_<0..4>.pth) -- that model was never trained
  on this dataset at all. Same probability-averaging as extract_inf_frames.py.

E.g. for a Paxos2020 frame: Paxos2020 fold-f model vs the 5-fold Paxos2025 ensemble.

Inside each mask the RGB pixels of the frame are shown, everything outside is black.
All panels are cropped to ONE common square window at native resolution (side =
largest bbox dimension of the masks, centered on their joint bbox, clamped to the
image) -- no resizing, no skewing. Predictions are mapped back to native frame
resolution (nearest neighbor) first.

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
    parser.add_argument("--n", type=int, default=25, help="frames (one PNG each) per eval dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 10})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transform(args.img_size)
    rng = random.Random(args.seed)

    for eval_ds in args.eval_datasets:
        cross = "2025" if eval_ds == "2020" else "2020"  # other cohort: its model never saw eval_ds
        df = pd.read_csv(Path(args.ann_dir) / f"annotations_cropping_{eval_ds}.csv").set_index("filename")
        with open(Path(args.ann_dir) / f"splits_{eval_ds}.json") as f:
            splits = json.load(f)
        candidates = [(fold, fn) for fold in range(args.n_splits) for fn in splits[f"fold_{fold}"]["val"]]
        picks = rng.sample(candidates, min(args.n, len(candidates)))

        # same-dataset fold models for the picked folds
        models = {}
        for fold, _ in picks:
            if (eval_ds, fold) not in models:
                path = Path(args.models_dir) / f"seg_{eval_ds}_fold_{fold}.pth"
                if not path.exists():
                    print(f"WARNING: missing {path}")
                    continue
                models[(eval_ds, fold)] = load_model(path, args.encoder_name, device)

        # cross-dataset ensemble: all 5 folds of the other cohort, mean sigmoid prob
        ens_models = []
        for efold in range(args.n_splits):
            epath = Path(args.models_dir) / f"seg_{cross}_fold_{efold}.pth"
            if epath.exists():
                ens_models.append(load_model(epath, args.encoder_name, device))
            else:
                print(f"WARNING: missing ensemble model {epath}")
        use_ensemble = len(ens_models) == args.n_splits
        if not use_ensemble:
            print(f"WARNING: ensemble incomplete ({len(ens_models)}/{args.n_splits}) -- ensemble panel skipped")

        vis_dir = out_dir / f"vis_compare_{eval_ds}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for fold, fn in picks:
            if (eval_ds, fold) not in models:
                continue
            img_bgr = cv2.imread(str(Path(args.img_dir) / fn))
            if img_bgr is None:
                print(f"WARNING: cannot read {args.img_dir}/{fn} -- skipped")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            gt = build_gt_mask(df.loc[fn], h, w)
            inp = transform(image=img)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                p_same = (torch.sigmoid(models[(eval_ds, fold)](inp))[0, 0] > 0.5).cpu().numpy()
            p_same = cv2.resize(p_same.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            ens = None
            if use_ensemble:
                with torch.no_grad():
                    probs = torch.stack([torch.sigmoid(m(inp)) for m in ens_models]).mean(dim=0)
                ep = (probs[0, 0] > 0.5).cpu().numpy()
                ens = cv2.resize(ep.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            window_masks = [gt, p_same] + ([ens] if ens is not None else [])
            r0, c0, side = common_square_window(window_masks, h, w)
            rgb = img[r0:r0 + side, c0:c0 + side]
            crops = {"gt": gt[r0:r0 + side, c0:c0 + side], "same": p_same[r0:r0 + side, c0:c0 + side]}
            if ens is not None:
                crops["ens"] = ens[r0:r0 + side, c0:c0 + side]

            panels = [("gt", "ground truth"),
                      ("same", f"model {eval_ds} (fold {fold}) | IoU {iou(crops['same'], crops['gt']):.3f}")]
            if ens is not None:
                panels.append(("ens", f"ensemble {cross} ({args.n_splits}-fold prob) | IoU {iou(crops['ens'], crops['gt']):.3f}"))

            fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6))
            axes[0].set_ylabel("\n".join(textwrap.wrap(fn, 34)) + f"\nfold {fold} | {side}px", fontsize=11)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            for j, (key, title) in enumerate(panels):
                ax = axes[j]
                masked = np.where(crops[key][..., None], rgb, 0)  # RGB inside mask, black outside
                ax.imshow(masked, interpolation="nearest")
                ax.set_title(title, fontsize=13, pad=8)
            fig.suptitle(f"Paxos{eval_ds} val frame: model {eval_ds} (fold {fold}, not trained on this split)"
                         + (f" vs {args.n_splits}-fold {cross} ensemble (mean sigmoid prob, never trained on {eval_ds})"
                            if ens is not None else "")
                         + f" — RGB inside mask, black outside; common {side}px square window, no resize",
                         fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = vis_dir / f"{n_written:02d}_{fn.replace(' ', '_')}"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            n_written += 1
        print(f"Wrote {n_written} per-frame plots to {vis_dir}")


if __name__ == "__main__":
    main()
