"""Tests for the additive monai_preproc package.

Run from the repo root:  python -m pytest tests/test_monai_preproc.py -q
All tests are CPU-only and use synthetic data (random frames, tiny synthetic
videos), so they run without the cluster, GPU or the real Paxos data.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from monai_preproc.configs import PreprocConfig, DEFAULT_GRID, load_grid
from monai_preproc.transforms import seg_val_transforms, clahe_np
from monai_preproc.models import (build_seg_model, build_cls_model, save_checkpoint,
                                  load_seg_model, load_cls_model, measure_cost)
from monai_preproc.train import train_seg, train_cls
from monai_preproc.evaluate import evaluate_seg, evaluate_cls
from monai_preproc.extract import main as extract_main, load_val_folds
from monai_preproc.sweep import pareto_front
from misc.utils import apply_clahe_cv2
from cropping.segmentation_pipeline import get_transforms as albu_get_transforms


def tiny_cfg(name="tiny"):
    return PreprocConfig(name=name, seg_arch="unet", seg_channels=(8, 16, 32), seg_strides=(2, 2),
                         seg_img_size=64, cls_backbone="efficientnet-b0", cls_img_size=64,
                         batch_size=2, cls_batch_size=2, epochs=1, cls_epochs=1)


def tiny_grid_file(tmp_path):
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({"tiny": tiny_cfg().to_dict()}))
    return str(p)


def random_frame(h=96, w=96, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, (255, 255, 255), -1)
    return img


def test_grid_loads(tmp_path):
    grid = load_grid(None)
    assert set(grid) == set(DEFAULT_GRID)
    override = tmp_path / "grid.json"
    override.write_text(json.dumps({"custom": {"seg_img_size": 256, "cls_backbone": "efficientnet-b0"}}))
    grid2 = load_grid(str(override))
    assert grid2["custom"].seg_img_size == 256
    assert "unet_tiny_256" in grid2


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


def test_clahe_parity_with_existing_utils():
    img = random_frame(80, 80, seed=2)
    ref = np.array(apply_clahe_cv2(Image.fromarray(img)))
    out = clahe_np(img.transpose(2, 0, 1).copy()).transpose(1, 2, 0)
    assert np.array_equal(out, ref)


def test_models_forward_and_cost():
    cfg = tiny_cfg()
    seg = build_seg_model(cfg)
    cls = build_cls_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert seg(x).shape == (2, 1, 64, 64)
        assert cls(x).shape == (2, 1)
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
    assert (models_dir / "seg_fold_0_log.json").exists()
    out_dir = tmp_path / "eval"
    summary = evaluate_seg(cfg, str(models_dir), str(ann_csv), str(splits_csv), str(img_dir),
                           str(out_dir), n_splits=1, batch_size=2, num_workers=0,
                           device=torch.device("cpu"), log=lambda *a: None)
    for f in ["per_image_metrics.csv", "summary.csv", "bar_summary.csv", "segmentation_eval.png"]:
        assert (out_dir / f).exists(), f
    assert {"micro_iou", "params", "flops", "latency_ms", "seg_img_size"} <= set(summary.columns)
    assert len(summary) == 1


def make_cls_fixture(tmp_path, n_per_label=3):
    img_base = tmp_path / "inf"
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    rows = []
    for label in (0, 1):
        d = img_base / "frames_cropped" / str(label)
        d.mkdir(parents=True)
        for i in range(n_per_label):
            name = f"fr{label}{i}.png"
            cv2.imwrite(str(d / name), random_frame(seed=100 + label * 10 + i))
            rows.append({"filename": name, "video_id": f"vid{label}{i}", "label": label,
                         "fold": 0, "split": "train" if i < n_per_label - 1 else "val"})
    (split_dir / "split_info_5fold_paxos2025_from_q5v3.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    return img_base, split_dir


def test_train_cls_and_evaluate_end_to_end(tmp_path):
    cfg = tiny_cfg()
    img_base, split_dir = make_cls_fixture(tmp_path)
    models_dir = tmp_path / "cls_models"
    res = train_cls(cfg, 0, str(img_base), str(split_dir), str(models_dir),
                    epochs=1, batch_size=2, num_workers=0, device=torch.device("cpu"), log=lambda *a: None)
    assert Path(res["model"]).exists()
    out_dir = tmp_path / "cls_eval"
    summary = evaluate_cls(cfg, str(models_dir), str(img_base), str(split_dir), str(out_dir),
                           n_splits=1, batch_size=2, num_workers=0, device=torch.device("cpu"),
                           log=lambda *a: None)
    for f in ["per_image_metrics.csv", "summary.csv", "bar_summary.csv", "classification_eval.png"]:
        assert (out_dir / f).exists(), f
    assert {"pr_auc", "params", "latency_ms", "cls_img_size"} <= set(summary.columns)


def make_video(path, n_frames=10, w=64, h=48, seed=0):
    rng = np.random.RandomState(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (w // 2, h // 2), min(w, h) // 4, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def make_extract_fixture(tmp_path):
    cfg = tiny_cfg()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    make_video(video_dir / "vidA.mp4", seed=5)
    (video_dir / "vidD.mp4").write_text("not a video")  # corrupt: on disk, unreadable
    q5_dir = tmp_path / "q5"
    for fold in range(1, 2):
        d = q5_dir / f"split_{fold}"
        d.mkdir(parents=True)
        (d / "mil_val_2025.json").write_text(json.dumps(
            {"vidA.mp4": 2, "vidB.mp4": 3, "vidC.mp4": 0, "vidD.mp4": 1}))
    seg_csv = tmp_path / "seg_splits.csv"
    cls_csv = tmp_path / "cls_splits.csv"
    pd.DataFrame([{"video_id": v, "fold": 0, "split": "val"} for v in ["vidA", "vidB", "vidC", "vidD"]]
                 ).to_csv(seg_csv, index=False)
    pd.DataFrame([{"video_id": v, "fold": 0, "split": "val"} for v in ["vidA", "vidB", "vidC", "vidD"]]
                 ).to_csv(cls_csv, index=False)
    seg_models = tmp_path / "seg_models"
    cls_models = tmp_path / "cls_models"
    save_checkpoint(build_seg_model(cfg), cfg, seg_models / "seg_fold_0.pth")
    save_checkpoint(build_cls_model(cfg), cfg, cls_models / "cls_fold_0.pth")
    return cfg, video_dir, q5_dir, seg_csv, cls_csv, seg_models, cls_models


def test_extract_skips_missing_and_corrupt_videos_and_matching_works(tmp_path, capsys):
    from data.dataset import MILVideoDataset
    cfg, video_dir, q5_dir, seg_csv, cls_csv, seg_models, cls_models = make_extract_fixture(tmp_path)
    out_base = tmp_path / "out"
    grid = tiny_grid_file(tmp_path)
    res = extract_main([
        "--video_dir", str(video_dir), "--q5_dir", str(q5_dir),
        "--seg_split_csv", str(seg_csv), "--cls_split_csv", str(cls_csv),
        "--seg_model_dir", str(seg_models), "--cls_model_dir", str(cls_models),
        "--out_base", str(out_base), "--n_splits", "1", "--infer_batch_size", "4",
        "--min_informative_frames", "3", "--grid", grid, "--config", "tiny",
    ])
    captured = capsys.readouterr().out
    assert "WARNING" in captured and "not present" in captured
    assert res["done"] == 1 and res["skipped"] == 1 and res["missing_locally"] == 2
    vid_dir = out_base / "cleaned_videos"
    assert (vid_dir / "CLEAN_vidA.mp4").exists()
    assert (out_base / "txt_files" / "vidA.txt").exists()
    assert (out_base / "plots" / "vidA.png").exists()
    assert not (vid_dir / "CLEAN_vidB.mp4").exists()

    label_json = tmp_path / "labels.json"
    label_json.write_text(json.dumps({"vidA.mp4": 2}))
    ds = MILVideoDataset(str(label_json), search_dir_paths=[str(vid_dir)])
    assert len(ds) == 1
    assert ds.labels == [1]
    assert re.search(r"vidA(?![a-zA-Z0-9])", "CLEAN_vidA.mp4")

    res2 = extract_main([
        "--video_dir", str(video_dir), "--q5_dir", str(q5_dir),
        "--seg_split_csv", str(seg_csv), "--cls_split_csv", str(cls_csv),
        "--seg_model_dir", str(seg_models), "--cls_model_dir", str(cls_models),
        "--out_base", str(out_base), "--n_splits", "1", "--infer_batch_size", "4",
        "--min_informative_frames", "3", "--grid", grid, "--config", "tiny",
    ])
    assert res2["done"] == 0 and res2["skipped"] == 1  # vidA outputs exist -> skipped; vidD still unreadable


def test_extract_rejects_protected_out_base(tmp_path):
    cfg, video_dir, q5_dir, seg_csv, cls_csv, seg_models, cls_models = make_extract_fixture(tmp_path)
    with pytest.raises(SystemExit):
        extract_main([
            "--video_dir", str(video_dir), "--q5_dir", str(q5_dir),
            "--seg_split_csv", str(seg_csv), "--cls_split_csv", str(cls_csv),
            "--seg_model_dir", str(seg_models), "--cls_model_dir", str(cls_models),
            "--out_base", "data/ensemble_results_paxos2025", "--n_splits", "1",
            "--grid", tiny_grid_file(tmp_path), "--config", "tiny",
        ])


def test_load_val_folds_warns_instead_of_asserting(tmp_path, capsys):
    csv = tmp_path / "sp.csv"
    pd.DataFrame([{"video_id": "vidX", "fold": 0, "split": "val"},
                  {"video_id": "vidX", "fold": 2, "split": "val"},
                  {"video_id": "vidY", "fold": 1, "split": "val"}]).to_csv(csv, index=False)
    mapping = load_val_folds(str(csv), log=lambda *a: None)
    assert mapping == {"vidX": 0, "vidY": 1}
    captured = capsys.readouterr().out
    assert captured == ""
    mapping2 = load_val_folds(str(csv))
    assert mapping2["vidX"] == 0


def test_pareto_front():
    rows = [{"perf": 0.9, "cost": 100.0},   # dominated by row 1 (same perf, lower cost)
            {"perf": 0.9, "cost": 50.0},    # pareto
            {"perf": 0.8, "cost": 10.0}]    # pareto (cheapest)
    idx = pareto_front(rows, "perf", "cost", maximize=True)
    assert set(idx) == {1, 2}
