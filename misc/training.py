"""Boilerplate shared by the training scripts (dino/train_dino_*.py, convnext/train_convnext_transformer.py)."""
import csv
import fcntl
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from misc.utils import save_confusion_matrix


def seed_everything(seed):
    """Best-effort reproducibility (weight init, sampler, dropout, frame sampling) -
    not bitwise-exact since cudnn's algorithm choice stays non-deterministic for speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch's default of one thread per core (80) oversubscribes the CPU when several runs share the
    # node (~3.5x slower). OMP_NUM_THREADS overrides the cap.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))
    cv2.setNumThreads(4)


def make_run_dir(model_dir, description, run_tag=None):
    """-> (run_name, run_dir): run_name = '<YYYY-MM-DD_HH-MM>_<description>[_<run_tag>]', run_dir = <model_dir>/models/<run_name>."""
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{description}{f'_{run_tag}' if run_tag else ''}"
    run_dir = os.path.join(model_dir, "models", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def video_search_dirs(fused):
    """cleaned_videos dirs of Paxos2025 (+ Paxos2020 if fused)."""
    dirs = ["data/ensemble_results_paxos2025/cleaned_videos"]
    if fused:
        dirs.append("data/ensemble_results_paxos2020/cleaned_videos")
    return dirs


def loader_kwargs(num_workers):
    """DataLoader worker settings shared by the training and eval loaders (decoding + augs overlap the GPU step).
    Workers run torch single-threaded by default; re-setting thread counts in a forked worker can deadlock."""
    if num_workers == 0:
        return {}
    return dict(num_workers=num_workers, persistent_workers=True)


def make_balanced_loaders(train_ds, val_ds, binary_classification, batch_size=1, require_all_classes=False,
                          num_workers=4):
    """Train loader oversampling by inverse class frequency (binary labels or 0-4 grades), plain val loader."""
    targets = np.array(train_ds.labels if binary_classification else train_ds.grades).astype(int)
    class_counts = np.bincount(targets)
    if require_all_classes:
        expected = 2 if binary_classification else 5
        assert len(class_counts) == expected and class_counts.all(), \
            f"Expected {expected} classes, got counts {class_counts.tolist()} - some class missing from split"
    sample_weights = torch.from_numpy((1.0 / class_counts)[targets]).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, **loader_kwargs(num_workers))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs(num_workers))
    return train_loader, val_loader


def make_optimizer(model, lr, backbone_lr_mult=None, optimizer="adam", weight_decay=0.01):
    """Adam/AdamW over the trainable params. With backbone_lr_mult, the backbone gets
    its own group at lr * backbone_lr_mult; param_groups[0] is always the head group."""
    if backbone_lr_mult is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    else:
        backbone_param_ids = {id(p) for p in model.backbone.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [{"params": head_params, "lr": lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    if optimizer == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    return optim.Adam(param_groups, lr=lr)


SELECTION_METRICS = ("default", "combined")

# One row per finished run (any script using EpochLogger): dataset/split, every CLI arg, best + last epoch metrics.
RESULTS_CSV = "results/run_results.csv"


def append_run_result(row, path=RESULTS_CSV):
    """Append row to the shared results CSV. Columns are the union over all rows (new args/metrics add
    columns, older rows get blanks); the file is locked so runs finishing at the same time don't collide."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, fieldnames = [], []
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        fieldnames += [k for k in row if k not in fieldnames]
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows + [row])
        os.replace(tmp, path)


def selection_score(m, binary_classification, selection="default"):
    """Checkpoint-selection score from a validate_* metrics dict. "default": PR-AUC (binary) / QWK (5-class);
    "combined" (5-class only): mean of QWK, referable recall and accuracy."""
    if selection == "combined":
        if binary_classification:
            raise ValueError("selection 'combined' is only defined for 5-class (QWK + referable recall + acc)")
        return (m['qwk'] + m['referable_recall'] + m['acc']) / 3
    return m['pr_auc'] if binary_classification else m['qwk']


class EpochLogger:
    """Per-epoch TensorBoard logging, confusion matrix, console line and best-checkpoint
    saving (selection metric: see selection_score)."""

    def __init__(self, run_name, run_dir, args, binary_classification, checkpoint_name, cm_to_tensorboard=False,
                 selection="default"):
        self.run_dir = run_dir
        self.binary = binary_classification
        self.checkpoint_path = os.path.join(run_dir, checkpoint_name)
        self.cm_to_tensorboard = cm_to_tensorboard
        self.selection = selection
        self.class_names = ['non-referable', 'referable'] if binary_classification else ['0', '1', '2', '3', '4']
        self.writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
        self.writer.add_text("args", str(args))
        # -inf (not 0.0) so a checkpoint is always written, even if the metric stays <= 0.
        self.best_selection_metric = float("-inf")
        self.best_metrics = None
        self.best_epoch = None
        self.last_metrics = None
        self.last_train_loss = None
        self.epochs_run = 0
        self.run_name = run_name
        self.args = args

    def log_epoch(self, epoch, model, optimizer, train_loss, m, extra_scalars=None):
        """m: metrics dict from a validate_* loop (incl. 'loss', 'y_true', 'y_pred')."""
        w = self.writer
        cm_path, _ = save_confusion_matrix(m['y_true'], m['y_pred'], epoch, self.run_dir, class_names=self.class_names)
        if self.cm_to_tensorboard:
            w.add_image('ConfusionMatrix', np.array(Image.open(cm_path).convert('RGB')).transpose(2, 0, 1), epoch)

        selection_metric = selection_score(m, self.binary, self.selection)
        self.last_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
        self.last_train_loss = train_loss
        self.epochs_run = epoch + 1
        w.add_scalar('Meta/Selection_Metric', selection_metric, epoch)
        if selection_metric > self.best_selection_metric:
            self.best_selection_metric = selection_metric
            self.best_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            w.add_scalar('Meta/Best_Selection_Metric', selection_metric, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        w.add_scalar('Meta/Learning_Rate', current_lr, epoch)
        if len(optimizer.param_groups) > 1:
            w.add_scalar('Meta/Backbone_Learning_Rate', optimizer.param_groups[1]['lr'], epoch)
        w.add_scalar('Loss/train', train_loss, epoch)
        w.add_scalar('Loss/val', m['loss'], epoch)
        for tag, value in (extra_scalars or {}).items():
            w.add_scalar(tag, value, epoch)
        # Every scalar metric, generically - covers binary and multiclass keys alike.
        for key, value in m.items():
            if key not in ('loss', 'y_true', 'y_pred'):
                w.add_scalar(f'Metric/{key}', value, epoch)

        headline_name, headline_value = ('PR-AUC', m['pr_auc']) if self.binary else ('QWK', m['qwk'])
        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {train_loss:.3f} | "
            f"Val Loss: {m['loss']:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {m['f1'] if self.binary else m['f1_macro']:.3f} | "
            f"{headline_name}: {headline_value:.3f}"
        )

    def results_row(self):
        """Shared-CSV row: run identity, dataset/split, all args (hp_*), best/last aggregate metrics (no per-class)."""
        args = vars(self.args)
        main_spec = getattr(sys.modules["__main__"], "__spec__", None)
        row = {
            "finished": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "run_name": self.run_name,
            "script": main_spec.name if main_spec else os.path.basename(sys.argv[0]),
            "run_dir": self.run_dir,
            "dataset": "paxos2025+paxos2020" if args.get("fused_dataset") else "paxos2025",
            "split": os.path.basename(os.path.normpath(args.get("split_path", ""))),
            "split_path": args.get("split_path"),
            "train_json": args.get("train_json"),
            "val_json": args.get("val_json"),
            "task": "binary" if self.binary else "5-class",
            "selection": self.selection,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_selection_metric": round(self.best_selection_metric, 4),
        }
        row.update({f"hp_{k}": v for k, v in args.items() if k not in ("split_path", "train_json", "val_json")})
        for prefix, metrics in (("best", self.best_metrics), ("last", self.last_metrics)):
            for k, v in (metrics or {}).items():
                if "_class" not in k:
                    row[f"{prefix}_{k}"] = round(float(v), 4)
        if self.last_train_loss is not None:
            row["last_train_loss"] = round(float(self.last_train_loss), 4)
        return row

    def close(self):
        self.writer.close()
        if self.epochs_run:
            append_run_result(self.results_row())
            print(f"Appended results to {RESULTS_CSV}")
        print(f"Done. Run directory: {self.run_dir}")
