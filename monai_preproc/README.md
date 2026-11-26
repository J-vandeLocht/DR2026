# MONAI pre-processing (additive)

A MONAI-based re-implementation of the Paxos2025 pre-processing pipeline. It is
**purely additive**: nothing in `cropping/`, `informative_frames/`, `data/` or
the existing `scripts/*.sh` is modified; the MONAI pipeline reads the *same*
split CSVs (`cropping/split_cropping_5fold_paxos2025_from_q5v3.csv`,
`informative_frames/split_info_5fold_paxos2025_from_q5v3.csv`, q5_v3 split
dirs), keeps the same anti-leakage val-fold protocol, and writes the same
output naming (`cleaned_videos/CLEAN_<video>.mp4`, `txt_files/<video>.txt`,
`plots/<video>.png`), so `data/match_clips.py`, `data/split_data.py` and
`data/dataset.py` matching keep working unchanged. Post-processing
(pad-to-square, largest connected component, square crop, CLAHE, frame
selection) is *imported* from `informative_frames/` — not copied — so the two
pipelines cannot drift apart.

## Layout

| File | Purpose |
|---|---|
| `configs.py` | `PreprocConfig` + named grids (`DEFAULT_GRID`, `FLEXUNET_GRID`) of configs (seg arch/size, cls backbone/size, seg shape-loss weight) for the cost/performance sweep |
| `losses.py` | `ellipse_shape_loss`: differentiable ellipse shape prior for the seg stage (matches soft moments of the prediction to the GT mask; for an ellipse the second moments give center + semi-axes) |
| `transforms.py` | MONAI transform chains; eval/inference chains are numerically equivalent to the albumentations/torchvision ones |
| `models.py` | MONAI `UNet` / `FlexibleUNet` (seg), MONAI `EfficientNetBN` (cls); checkpoint + config-sidecar I/O; `measure_cost` (params/FLOPs/latency) |
| `data.py` | Datasets over the existing annotation/split CSVs; **rows whose files are missing on disk are skipped with a warning, never an error** |
| `train.py` | `python -m monai_preproc.train seg|cls --config NAME --fold i` (best val IoU / PR-AUC, `seg_fold_<i>.pth` / `cls_fold_<i>.pth` + `.json` sidecar + `_log.json`) |
| `evaluate.py` | `python -m monai_preproc.evaluate seg|cls ...` — same output CSVs/plot as `cropping/evaluate_segmentation.py` / `informative_frames/evaluate_classification.py`, plus cost columns in `summary.csv` |
| `extract.py` | `python -m monai_preproc.extract ...` — videos → `CLEAN_<video>.mp4` + txt + plot; q5_v3 videos missing/unreadable locally are skipped with a warning; `--out_base data/ensemble_results_paxos2025` is rejected like in the existing extractor |
| `sweep.py` | `python -m monai_preproc.sweep --out_dir results/monai_sweep [--preset flexunet] [--stage seg|cls|both]` — trains+evaluates the selected stage(s) of every config in the grid, writes `sweep_results.csv`, `pareto.csv` and the overview plot `overview.png` |

Shell entry points (additive): `scripts/monai_preprocess_all.sh` (full pipeline,
`CONFIG=` selects the grid entry) and `scripts/monai_sweep.sh` (sweep).

## Quick start

```bash
pip install monai   # plus the repo's existing env job_new

# full MONAI pipeline (one config), mirrors scripts/preprocess_all.sh
CONFIG=unet_small_384 sbatch scripts/monai_preprocess_all.sh

# just extraction with trained MONAI models
python -u -m monai_preproc.extract --video_dir data/dr_videos \
  --seg_model_dir monai_preproc/models/seg --cls_model_dir monai_preproc/models/cls \
  --config unet_small_384 --out_base data/ensemble_results_paxos2025_monai

# cost/performance sweep over the FlexibleUNet architecture grid
sbatch scripts/monai_sweep.sh            # or: python -u -m monai_preproc.sweep --out_dir results/monai_sweep
```

## Cost/performance sweep

Two named grids live in `configs.py` (`GRIDS`): `default` (`DEFAULT_GRID`, the
original mixed UNet-vs-FlexibleUNet comparison) and `flexunet`
(`FLEXUNET_GRID`, the default of `monai_preproc.sweep` / `scripts/monai_sweep.sh`:
FlexibleUNet with EfficientNet-B0…B4 backbones at 256–768 input, the only
backbones with ImageNet weights in 2D — the sweep showed plain UNet encoders
cannot match them on the fundus crop mask). All grid configs use
ImageNet-pretrained backbones (`pretrained=True`), matching the old pipeline's
`--encoder_weights imagenet` / `EfficientNet_B4_Weights.DEFAULT`; checkpoints
are always *loaded* with `pretrained=False` so no weights are downloaded at
eval/extract time. Each config is trained on all 5 folds and evaluated on the
val folds (never trained on), with per-frame latency/params/FLOPs recorded.

The two pipeline stages are searched separately: run `--stage seg` first to
pick the segmentation architecture, then `--stage cls` for the
informative-frame classifier (`STAGE=seg sbatch scripts/monai_sweep.sh`,
`--no-train` re-evaluates existing checkpoints). Outputs in `--out_dir`
(default `results/monai_sweep`):

- `sweep_results.csv` — one row per config+stage (`perf` = mean micro IoU for
  seg, mean PR-AUC for cls; cost = `latency_ms` per frame, `params`, `flops`)
- `pareto.csv` — Pareto-optimal configs per stage
- `overview.png` — performance-vs-cost plot per stage with the Pareto front
  highlighted (re-render from an existing CSV without training:
  `python -u -m monai_preproc.sweep --plot_only results/monai_sweep/sweep_results.csv`)
- `models/<config>/{seg,cls}/`, `eval/<config>/{seg,cls}/` — checkpoints and
  per-fold eval detail (same CSV/plot format as `cropping/eval/` and
  `informative_frames/eval/`)

Extend or override a grid with a JSON file:
`python -m monai_preproc.sweep --grid my_grid.json`, where the file is
`{"my_config": {"seg_img_size": 384, "cls_backbone": "efficientnet-b0", ...}}`;
`train.py` / `evaluate.py` / `extract.py` take the same `--preset`/`--grid`
flags to address configs by name.

### Shape-prior loss for the segmentation stage

The crop target is always an axis-aligned ellipse (the annotation CSV stores
`center_x/center_y/radius_x/radius_y`), so the seg loss can exploit that:
`losses.py` adds `ellipse_shape_loss` — the MSE between the soft ellipse
parameters of the predicted probability map and those of the GT mask (for a
filled ellipse, `a = 2*sqrt(var_x)`, `b = 2*sqrt(var_y)` in normalized
coordinates, so matching moments = matching center and radii). Training uses
`DiceCELoss + seg_shape_loss_weight * ellipse_shape_loss`; weight `0` is exactly
the previous loss.

To compare old vs new loss in one sweep, pass `--shape_loss W`
(`SHAPE_LOSS=100 sbatch scripts/monai_sweep.sh` with `STAGE=seg`): every config
is additionally run as a seg-only variant named `<config>_shapeW`. The variants
appear as separate rows in `sweep_results.csv`/`pareto.csv` and in orange in
`overview.png`, so the better loss is directly visible per architecture. Set the
weight per config with `"seg_shape_loss_weight": W` in a `--grid` JSON.

## Robustness notes

- Videos/frames listed in the q5_v3 splits or annotation CSVs but not present
  locally are skipped with a WARNING and counted, never an error
  (`extract.py`, `data.py`).
- Videos in the val part of several folds (can happen when new videos are
  added) get the lowest fold with a WARNING instead of an assertion.
- `--out_base data/ensemble_results_paxos2025` is rejected (same guard as the
  existing extractor) so re-runs never overwrite the in-use clips.

## Tests

```bash
python -m pytest tests/test_monai_preproc.py -q
```

Covers transform parity with the existing albumentations chains, model forward
passes, end-to-end seg train+eval on synthetic frames, extraction on synthetic
videos including missing/unreadable videos, and clip matching via
`data.dataset.MILVideoDataset` on the produced `CLEAN_*` files.
