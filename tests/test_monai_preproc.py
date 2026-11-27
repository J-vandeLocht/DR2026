"""
Run from the repo root:  python -m pytest tests/test_monai_preproc.py -q
All tests are CPU-only and use synthetic data (random frames), so they run
without the cluster, GPU or the real Paxos data.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from monai_preproc.configs import PreprocConfig, DEFAULT_GRID, FLEXUNET_GRID, load_grid
from monai_preproc.transforms import seg_val_transforms
from monai_preproc.models import build_seg_model, save_checkpoint, load_seg_model, measure_cost
from monai_preproc.train import train_seg
from monai_preproc.evaluate import evaluate_seg
from monai_preproc.losses import ellipse_moments, ellipse_shape_loss
from monai_preproc.sweep import pareto_front, pareto_per_stage
from cropping.segmentation_pipeline import get_transforms as albu_get_transforms


def tiny_cfg(name="tiny"):
    return PreprocConfig(name=name, seg_arch="unet", seg_channels=(8, 16, 32), seg_strides=(2, 2),
                         seg_img_size=64, batch_size=2, epochs=1, patience=3)


def random_frame(h=96, w=96, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, (255, 255, 255), -1)
    return img


def test_grid_loads(tmp_path):
    grid = load_grid(None)
    assert set(grid) == set(DEFAULT_GRID)
    override = tmp_path / "grid.json"
    override.write_text(json.dumps({"custom": {"seg_img_size": 256, "seg_backbone": "efficientnet-b0"}}))
    grid2 = load_grid(str(override))
    assert grid2["custom"].seg_img_size == 256
    assert "unet_tiny_256" in grid2
    flex = load_grid(None, preset="flexunet")
    assert set(flex) == set(FLEXUNET_GRID)
    assert all(c.seg_arch == "flexunet" for c in flex.values())
    assert all(c.pretrained for c in flex.values())


def test_grid_unet_strides_match_channels():
    # MONAI UNet expects len(strides) == len(channels) - 1 (extra strides are
    # ignored with a UserWarning); the grid must not trigger that warning
    for cfg in DEFAULT_GRID.values():
        if cfg.seg_arch == "unet":
            assert len(cfg.seg_strides) == len(cfg.seg_channels) - 1


def test_seg_val_transform_parity_with_albumentations():
    img = random_frame(100, 60, seed=1)
    mask = np.zeros((100, 60), dtype=np.float32)
    cv2.ellipse(mask, (30, 50), (20, 25), 0, 0, 360, 1.0, -1)
    albu_t = albu_get_transforms(train=False, img_size=64)
    out_albu = albu_t(image=img, mask=mask)
    monai_t = seg_val_transforms(64)
    out_monai = monai_t({"image": img, "mask": mask})
    assert out_monai["image"].shape == (3, 64, 64)
    assert out_monai["mask"].shape == (1, 64, 64)
    assert torch.allclose(out_monai["image"], out_albu["image"], atol=2e-2)
    assert torch.allclose(out_monai["mask"][0], out_albu["mask"], atol=1e-6)


def test_models_forward_and_cost():
    cfg = tiny_cfg()
    seg = build_seg_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert seg(x).shape == (2, 1, 64, 64)
    cost = measure_cost(seg, 64, torch.device("cpu"))
    assert cost["params"] > 0 and cost["latency_ms"] > 0 and cost["img_size"] == 64


def test_checkpoint_roundtrip_with_sidecar(tmp_path):
    cfg = tiny_cfg()
    model = build_seg_model(cfg)
    ckpt = tmp_path / "seg_fold_0.pth"
    save_checkpoint(model, cfg, ckpt)
    assert ckpt.with_suffix(".json").exists()
    loaded = load_seg_model(ckpt, torch.device("cpu"))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(loaded(x), model.eval()(x))


def make_seg_fixture(tmp_path, n_train=4, n_val=2):
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    rows, split_rows = [], []
    for i in range(n_train + n_val):
        name = f"f{i:03d}.png"
        cv2.imwrite(str(img_dir / name), random_frame(seed=10 + i))
        rows.append({"filename": name, "center_x": 48, "center_y": 48,
                     "radius_x": 24, "radius_y": 24})
        split_rows.append({"filename": name, "video_id": f"vid{i}", "fold": 0,
                           "split": "train" if i < n_train else "val"})
    ann_csv = tmp_path / "ann.csv"
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(ann_csv, index=False)
    pd.DataFrame(split_rows).to_csv(splits_csv, index=False)
    return ann_csv, splits_csv, img_dir


def test_train_seg_and_evaluate_end_to_end(tmp_path):
    cfg = tiny_cfg()
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    models_dir = tmp_path / "models"
    res = train_seg(cfg, 0, str(ann_csv), str(splits_csv), str(img_dir), str(models_dir),
                    epochs=1, batch_size=2, num_workers=0, device=torch.device("cpu"), log=lambda *a: None)
    assert Path(res["model"]).exists()
    assert (models_dir / "seg_fold_0.json").exists()
    assert not (models_dir / "seg_fold_0_log.json").exists()  # TensorBoard instead of json logs
    assert list((models_dir / "runs" / "fold_0").glob("events.*")), "tensorboard events missing"
    assert list((models_dir / "vis").glob("fold0_best_epoch*.png")), "best-IoU 2x5 plot missing"
    out_dir = tmp_path / "eval"
    summary = evaluate_seg(cfg, str(models_dir), str(ann_csv), str(splits_csv), str(img_dir),
                           str(out_dir), n_splits=1, batch_size=2, num_workers=0,
                           device=torch.device("cpu"), log=lambda *a: None)
    for f in ["per_image_metrics.csv", "summary.csv", "bar_summary.csv", "segmentation_eval.png"]:
        assert (out_dir / f).exists(), f
    assert {"micro_iou", "macro_iou", "params", "flops", "latency_ms", "seg_img_size"} <= set(summary.columns)
    assert len(summary) == 1


def test_train_seg_early_stopping(tmp_path, monkeypatch):
    import monai_preproc.train as train_mod
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    monkeypatch.setattr(train_mod, "binary_counts",
                        lambda pred, target: tuple(torch.zeros(pred.shape[0], dtype=torch.long) for _ in range(4)))
    cfg = tiny_cfg()
    res = train_mod.train_seg(cfg, 0, str(ann_csv), str(splits_csv), str(img_dir), str(tmp_path / "m"),
                              epochs=5, patience=2, num_workers=0, device=torch.device("cpu"),
                              log=lambda *a: None)
    # val IoU is 0 every epoch: epoch 0 improves over -1, then patience=2 stops at epoch 2
    assert res["epochs_run"] == 3
    assert res["best_epoch"] == 0
    assert Path(res["model"]).exists()


def test_train_seg_early_stopping_disabled_with_patience_0(tmp_path, monkeypatch):
    import monai_preproc.train as train_mod
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    monkeypatch.setattr(train_mod, "binary_counts",
                        lambda pred, target: tuple(torch.zeros(pred.shape[0], dtype=torch.long) for _ in range(4)))
    cfg = tiny_cfg()
    res = train_mod.train_seg(cfg, 0, str(ann_csv), str(splits_csv), str(img_dir), str(tmp_path / "m"),
                              epochs=5, patience=0, num_workers=0, device=torch.device("cpu"),
                              log=lambda *a: None)
    assert res["epochs_run"] == 5  # patience=0 disables early stopping


def test_vis_seg_fold_writes_png_and_csv(tmp_path):
    from monai_preproc.vis_seg import vis_seg_fold
    cfg = tiny_cfg()
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    models_dir = tmp_path / "models"
    train_seg(cfg, 0, str(ann_csv), str(splits_csv), str(img_dir), str(models_dir),
              epochs=1, batch_size=2, num_workers=0, device=torch.device("cpu"), log=lambda *a: None)
    out_dir = tmp_path / "vis"
    png, csv = vis_seg_fold(cfg, 0, models_dir, str(ann_csv), str(splits_csv), str(img_dir),
                            str(out_dir), n_images=2, batch_size=2, num_workers=0,
                            device=torch.device("cpu"), log=lambda *a: None)
    assert png.exists() and csv.exists()
    assert len(pd.read_csv(csv)) == 2


def test_pareto_front():
    rows = [{"perf": 0.9, "cost": 100.0},   # dominated by row 1 (same perf, lower cost)
            {"perf": 0.9, "cost": 50.0},    # pareto
            {"perf": 0.8, "cost": 10.0}]    # pareto (cheapest)
    idx = pareto_front(rows, "perf", "cost", maximize=True)
    assert set(idx) == {1, 2}


def test_pareto_per_stage_keeps_full_frame_rows():
    # run_sweep appends one row per config; the Pareto selection per stage must
    # map back to the original df rows
    df = pd.DataFrame([
        {"config": "a", "stage": "seg", "perf": 0.9, "params": 100, "latency_ms": 20.0},
        {"config": "a", "stage": "cls", "perf": 0.95, "params": 50, "latency_ms": 10.0},
        {"config": "b", "stage": "seg", "perf": 0.8, "params": 10, "latency_ms": 2.0},
        {"config": "b", "stage": "cls", "perf": 0.96, "params": 5, "latency_ms": 1.0},
    ])
    pareto = pareto_per_stage(df)
    assert set(zip(pareto["config"], pareto["stage"])) == {("a", "seg"), ("b", "seg"), ("b", "cls")}


def test_ellipse_moments_recover_ellipse_and_shape_loss_gradients():
    mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
    cv2.ellipse(mask[0, 0], (32, 32), (16, 16), 0, 0, 360, 1.0, -1)
    m = ellipse_moments(torch.from_numpy(mask))
    assert m.shape == (1, 4)
    # normalized coords: pixel 32 -> 32/63, semi-axis 16 px -> 16/63 (a = 2*sqrt(var) = semi-axis)
    assert torch.allclose(m[0], torch.tensor([32 / 63, 32 / 63, 16 / 63, 16 / 63]), atol=0.02)
    logits = (torch.from_numpy(mask) - 0.5) * 20.0
    assert ellipse_shape_loss(logits, torch.from_numpy(mask)).item() < 1e-3  # perfect pred -> ~0
    shifted = np.zeros_like(mask)
    cv2.ellipse(shifted[0, 0], (40, 32), (16, 16), 0, 0, 360, 1.0, -1)
    assert ellipse_shape_loss(logits, torch.from_numpy(shifted)).item() > 1e-3  # wrong center -> >0
    logits = logits.clone().requires_grad_(True)
    ellipse_shape_loss(logits, torch.from_numpy(shifted)).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_sweep_writes_csv_pareto_and_overview_plot(tmp_path):
    from monai_preproc.sweep import run_sweep
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    grid = {"tiny": tiny_cfg()}  # single-entry grid: load_grid(json) would extend the whole preset
    out_dir = tmp_path / "sweep"
    df = run_sweep(grid, str(out_dir), folds=[0], epochs=1, train=True,
                   ann_csv=str(ann_csv), splits_csv=str(splits_csv), img_dir=str(img_dir),
                   num_workers=0, device=torch.device("cpu"), log=lambda *a: None)
    assert set(df["stage"]) == {"seg"}
    assert (out_dir / "sweep_results.csv").exists()
    assert (out_dir / "pareto.csv").exists()
    assert (out_dir / "overview.png").exists()
    assert (out_dir / "models" / "tiny" / "seg" / "seg_fold_0.pth").exists()


def test_sweep_shape_loss_adds_variants(tmp_path):
    from monai_preproc.sweep import run_sweep
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    grid = {"tiny": tiny_cfg()}  # single-entry grid: load_grid(json) would extend the whole preset
    out_dir = tmp_path / "sweep"
    df = run_sweep(grid, str(out_dir), folds=[0], epochs=1, train=True,
                   ann_csv=str(ann_csv), splits_csv=str(splits_csv), img_dir=str(img_dir),
                   num_workers=0, device=torch.device("cpu"), seg_shape_weights=(100.0,),
                   log=lambda *a: None)
    assert set(df["config"]) == {"tiny", "tiny_shape100"}
    assert set(df["stage"]) == {"seg"}
    assert (out_dir / "models" / "tiny_shape100" / "seg" / "seg_fold_0.pth").exists()
    assert (out_dir / "overview.png").exists()


def test_sweep_shape_variants_only(tmp_path):
    from monai_preproc.sweep import run_sweep
    ann_csv, splits_csv, img_dir = make_seg_fixture(tmp_path)
    grid = {"tiny": tiny_cfg()}
    out_dir = tmp_path / "sweep"
    df = run_sweep(grid, str(out_dir), folds=[0], epochs=1, train=True,
                   ann_csv=str(ann_csv), splits_csv=str(splits_csv), img_dir=str(img_dir),
                   num_workers=0, device=torch.device("cpu"), seg_shape_weights=(100.0,),
                   shape_variants_only=True, log=lambda *a: None)
    assert set(df["config"]) == {"tiny_shape100"}
    assert not (out_dir / "models" / "tiny").exists()


def test_sweep_plot_only_rerenders_outputs(tmp_path):
    from monai_preproc.sweep import main as sweep_main
    df = pd.DataFrame([
        {"config": "a", "stage": "seg", "perf": 0.9, "params": 100, "flops": None, "latency_ms": 20.0, "img_size": 64, "arch": "unet"},
        {"config": "a", "stage": "cls", "perf": 0.95, "params": 50, "flops": None, "latency_ms": 10.0, "img_size": 64, "arch": "efficientnet-b0"},
        {"config": "b", "stage": "seg", "perf": 0.8, "params": 10, "flops": None, "latency_ms": 2.0, "img_size": 64, "arch": "unet"},
        {"config": "b", "stage": "cls", "perf": 0.96, "params": 5, "flops": None, "latency_ms": 1.0, "img_size": 64, "arch": "efficientnet-b0"},
    ])
    csv = tmp_path / "sweep_results.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out"
    res = sweep_main(["--plot_only", str(csv), "--out_dir", str(out)])
    assert len(res) == 4
    assert (out / "pareto.csv").exists() and (out / "overview.png").exists()
    pareto = pd.read_csv(out / "pareto.csv")
    assert set(zip(pareto["config"], pareto["stage"])) == {("a", "seg"), ("b", "seg"), ("b", "cls")}
