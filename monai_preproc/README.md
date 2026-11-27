# MONAI segmentation (additive)

A MONAI-based re-implementation of the Paxos2025 **cropping/segmentation** stage.
It is **purely additive**: nothing in `cropping/`, `informative_frames/`, `data/`
or the existing `scripts/*.sh` is modified; the MONAI pipeline reads the *same*
split CSV (`cropping/split_cropping_5fold_paxos2025_from_q5v3.csv`) and keeps the
same anti-leakage val-fold protocol as `cropping/train_segmentation.py` (the
original smp `UnetPlusPlus`/EfficientNet-B4 pipeline): fold *i* trains on
`fold==i & split=="train"` and validates on `fold==i & split=="val"`.

The informative-frame **classifier** is a separate pipeline and is intentionally
not part of this package for now; it will be re-added once the best seg config
is found.

## How it mirrors the original UNet++ pipeline

- Model: `FlexibleUNet` with an ImageNet-pretrained EfficientNet backbone
  (`flexunet_b4_768` matches `smp.UnetPlusPlus(efficientnet-b4, imagenet)` @768),
  or a plain MONAI `UNet` (`unet_*` configs).
- Training (`train.py`): Adam(lr=1e-4), batch_size=4, epochs=40, Dice loss from
  logits with smooth=1.0 — the same form as `smp.losses.DiceLoss(BINARY_MODE,
  from_logits=True)`; optional AMP (`--amp`); seed 42; best-val-micro-IoU
  checkpoint `seg_fold_<fold>.pth` (+ `.json` config sidecar), same naming as
  `cropping/models/`.
- Transforms (`transforms.py`): the val chain is numerically identical to
  `cropping.segmentation_pipeline.get_transforms(train=False)` (parity-tested);
  the train chain mirrors `get_transforms(train=True)` (random 80–100% square
  crop, flips, affine rotate ±180°/translate ±5%/scale 0.95–1.05, blur,
  brightness/contrast, ImageNet normalize).
- Evaluation (`evaluate.py`): per-image IoU/Dice/accuracy at threshold 0.5, micro
  + macro per fold; outputs (`per_image_metrics.csv`, `summary.csv`,
  `bar_summary.csv`, `segmentation_eval.png`) match `cropping/eval/` exactly,
  plus cost columns (params/flops/latency_ms) in `summary.csv`.

On top of the original, training adds **TensorBoard logging**
(`<out_dir>/runs/fold_<i>/`; `tensorboard --logdir <out_dir>/runs`) with the loss
components logged separately (`train/loss`, `train/loss_dice`, `train/loss_bce`,
`val/iou`, `best/val_iou`, plus the improvement figure), **early
stopping** (`patience=3` by default; disable with `PATIENCE=0 sbatch
scripts/monai_sweep.sh` or `--patience 0`), and a **2×5 visualization saved on every
new best val IoU** (`<out_dir>/vis/fold<i>_best_epoch<e>.png`: top row = original
val images, bottom row = image with everything outside the predicted mask
blacked out; also logged to TensorBoard as `val/best_masked`).

## Layout

| File | Purpose |
|---|---|
| `configs.py` | `PreprocConfig` + named grids (`DEFAULT_GRID`, `FLEXUNET_GRID`) for the cost/performance sweep |
| `transforms.py` | MONAI transform chains (val chain parity-tested against albumentations) |
| `models.py` | MONAI `UNet` / `FlexibleUNet`; checkpoint + config-sidecar I/O; `measure_cost` (params/FLOPs/latency) |
| `data.py` | Dataset over the cropping annotation/split CSVs; **missing files are skipped with a warning, never an error** |
| `train.py` | `python -m monai_preproc.train --config NAME --fold i` — TensorBoard + early stopping + best-IoU checkpoint and 2×5 plot |
| `evaluate.py` | `python -m monai_preproc.evaluate --config NAME --models_dir ... --out_dir ...` — same outputs as `cropping/eval/` + cost columns |
| `sweep.py` | `python -m monai_preproc.sweep --out_dir results/monai_sweep` — trains+evaluates every grid config, writes `sweep_results.csv`, `pareto.csv`, `overview.png` |
| `vis_seg.py` | `python -m monai_preproc.vis_seg --config NAME --folds 0 1` — per-fold val prediction figures (input | GT | pred | TP/FP/FN + fitted ellipses) |

Shell entry points (additive): `scripts/monai_sweep.sh` (sweep) and
`scripts/monai_vis_seg.sh` (val-prediction plots).

## Quick start

```bash
pip install monai tensorboard   # plus the repo's existing env job_new

# train one config, one fold (TensorBoard logs under monai_preproc/models/seg/runs/)
python -u -m monai_preproc.train --config flexunet_b4_768 --fold 0 --amp
tensorboard --logdir monai_preproc/models/seg/runs

# evaluate it (same output files as cropping/eval/)
python -u -m monai_preproc.evaluate --config flexunet_b4_768 \
  --models_dir monai_preproc/models/seg --out_dir monai_preproc/eval/flexunet_b4_768

# inspect val predictions of a swept config
sbatch scripts/monai_vis_seg.sh    # CONFIG=flexunet_b0_256 FOLDS="0 1" N_IMAGES=6 sbatch ...

# cost/performance sweep over the FlexibleUNet grid
sbatch scripts/monai_sweep.sh      # or: python -u -m monai_preproc.sweep --out_dir results/monai_sweep
```

## Cost/performance sweep

Two named grids live in `configs.py` (`GRIDS`): `default` (`DEFAULT_GRID`, the
original mixed UNet-vs-FlexibleUNet comparison) and `flexunet` (`FLEXUNET_GRID`,
the default of `monai_preproc.sweep` / `scripts/monai_sweep.sh`: FlexibleUNet
with EfficientNet-B0…B4 backbones at 256–768 input — the only backbones with
ImageNet weights in 2D; the first sweep showed plain UNet encoders cannot match
them on the fundus crop mask). All grid configs use ImageNet-pretrained
backbones (`pretrained=True`), matching the old pipeline's
`encoder_weights="imagenet"`; checkpoints are always *loaded* with
`pretrained=False` so no weights are downloaded at eval time. Each config is
trained on all 5 folds and evaluated on the val folds (never trained on), with
per-frame latency/params/FLOPs recorded.

Outputs in `--out_dir` (default `results/monai_sweep`):

- `sweep_results.csv` — one row per config (`perf` = mean micro IoU over the 5
  val folds; cost = `latency_ms` per frame, `params`, `flops`)
- `pareto.csv` — Pareto-optimal configs
- `overview.png` — performance-vs-cost plot with the Pareto front highlighted
  (re-render from an existing CSV without training:
  `python -u -m monai_preproc.sweep --plot_only results/monai_sweep/sweep_results.csv`)
- `models/<config>/seg/` — fold checkpoints, TensorBoard runs (`runs/`), and the
  best-IoU 2×5 plots (`vis/`)
- `eval/<config>/seg/` — per-fold eval detail (same CSV/plot format as `cropping/eval/`)

### Loss comparison (Dice vs Dice+CE)

Only stock losses are used — no custom loss. The default `seg_loss="dice"` is
exactly the original pipeline's loss; `seg_loss="dice_ce"` adds plain BCE on top
(the stock MONAI `DiceCELoss` combination), which helps because most pixels are
background. To compare in one sweep: `LOSS_VARIANT=dice_ce sbatch
scripts/monai_sweep.sh` — every config is additionally run as `<config>_dicece`,
shown as separate rows in the CSVs and in orange in `overview.png`. To run *only*
the variants, add `ONLY_VARIANT=1`. In TensorBoard both components are logged
separately (`train/loss_dice`, `train/loss_bce`, plus the total `train/loss`), so
the two losses can be inspected independently per run.

Extend or override a grid with a JSON file:
`python -m monai_preproc.sweep --grid my_grid.json`, where the file is
`{"my_config": {"seg_img_size": 384, "seg_backbone": "efficientnet-b0", ...}}`;
`train.py` / `evaluate.py` / `vis_seg.py` take the same `--preset`/`--grid` flags
to address configs by name.

## Tests

```bash
python -m pytest tests/test_monai_preproc.py -q
```

Covers val-transform parity with the albumentations chain, model forward + cost,
checkpoint roundtrip, end-to-end seg train (TensorBoard events, early stopping,
best-IoU plot, separate loss-component logging) + eval on synthetic frames, and
the sweep (CSV/pareto/overview outputs, loss variants, `--plot_only`).
