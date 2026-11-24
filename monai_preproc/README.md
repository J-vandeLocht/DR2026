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
| `configs.py` | `PreprocConfig` + `DEFAULT_GRID` of named configs (seg arch/size, cls backbone/size) for the cost/performance sweep |
| `transforms.py` | MONAI transform chains; eval/inference chains are numerically equivalent to the albumentations/torchvision ones |
| `models.py` | MONAI `UNet` / `FlexibleUNet` (seg), MONAI `EfficientNetBN` (cls); checkpoint + config-sidecar I/O; `measure_cost` (params/FLOPs/latency) |
| `data.py` | Datasets over the existing annotation/split CSVs; **rows whose files are missing on disk are skipped with a warning, never an error** |
| `train.py` | `python -m monai_preproc.train seg|cls --config NAME --fold i` (best val IoU / PR-AUC, `seg_fold_<i>.pth` / `cls_fold_<i>.pth` + `.json` sidecar + `_log.json`) |
| `evaluate.py` | `python -m monai_preproc.evaluate seg|cls ...` — same output CSVs/plot as `cropping/evaluate_segmentation.py` / `informative_frames/evaluate_classification.py`, plus cost columns in `summary.csv` |
| `extract.py` | `python -m monai_preproc.extract ...` — videos → `CLEAN_<video>.mp4` + txt + plot; q5_v3 videos missing/unreadable locally are skipped with a warning; `--out_base data/ensemble_results_paxos2025` is rejected like in the existing extractor |
| `sweep.py` | `python -m monai_preproc.sweep --out_dir results/monai_sweep` — trains+evaluates every config, writes `sweep_results.csv` and `pareto.csv` (best cost/performance trade-off) |

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

# cost/performance sweep over all grid configs
sbatch scripts/monai_sweep.sh            # or: python -u -m monai_preproc.sweep --out_dir results/monai_sweep
```

## Cost/performance sweep

`DEFAULT_GRID` in `configs.py` spans a baseline (FlexibleUNet/EfficientNet-B4
@ 768/512, matching the current smp pipeline's capacity) down to small UNet /
EfficientNet-B0 @ 256–384. Each config is trained on all 5 folds and evaluated
on the val folds (never trained on), with per-frame latency/params/FLOPs
recorded; `sweep.py` then reports the Pareto front in `pareto.csv`. Extend or
override the grid with a JSON file: `python -m monai_preproc.sweep --grid my_grid.json`,
where the file is `{"my_config": {"seg_img_size": 384, "cls_backbone": "efficientnet-b0", ...}}`.

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
