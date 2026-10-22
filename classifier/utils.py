import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from misc.utils import binary_metrics, multiclass_metrics

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, cohen_kappa_score,
                             precision_score, recall_score, average_precision_score)


def train_one_epoch_classifier(model, loader, criterion, optimizer, binary_classification, device,
                               distance_scale_weight=0.0):
    """distance_scale_weight > 0 scales each frame's CE by 1 + w*|argmax - grade| (5-class only, as in
    transformer/utils.py::train_one_epoch_trans)."""
    model.train()
    total_loss = 0

    for images, grades, labels in tqdm(loader, leave=False, desc="Batch"):
        images = images.to(device)
        if binary_classification:
            targets = labels.to(device).float().unsqueeze(1)   # BCEWithLogitsLoss wants (N,1) float
        else:
            targets = grades.to(device).long()                  # CrossEntropyLoss wants (N,) long

        optimizer.zero_grad()
        outputs = model(images)
        if distance_scale_weight > 0 and not binary_classification:
            with torch.no_grad():
                scale = 1.0 + distance_scale_weight * (outputs.argmax(dim=-1) - targets).abs().float()
            loss = (F.cross_entropy(outputs, targets, reduction="none") * scale).mean()
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_classifier(model, loader, criterion, binary_classification, device):
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []

    with torch.no_grad():
        for images, grades, labels in loader:
            images = images.to(device)
            if binary_classification:
                targets = labels.to(device).float().unsqueeze(1)
            else:
                targets = grades.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            if binary_classification:
                probs = torch.sigmoid(outputs)          # (N,1)
            else:
                probs = torch.softmax(outputs, dim=1)    # (N,5)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    y_true = np.array(y_true).reshape(-1) if binary_classification else np.array(y_true)
    y_probs = np.array(y_probs)

    if binary_classification:
        y_pred = (y_probs.reshape(-1) > 0.5).astype(float)
        metrics = binary_metrics(y_true, y_pred, y_probs.reshape(-1))
    else:
        y_pred = y_probs.argmax(axis=1)
        metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes=5)

    metrics['loss'] = running_loss / len(loader)
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    return metrics
