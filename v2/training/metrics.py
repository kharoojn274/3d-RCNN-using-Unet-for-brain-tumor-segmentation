from __future__ import annotations

import numpy as np
import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False, eps: float = 1e-8) -> float:
    pred = torch.argmax(logits, dim=1)
    classes = range(logits.shape[1]) if include_background else range(1, logits.shape[1])

    scores = []
    for c in classes:
        p = pred == c
        t = target == c
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            continue
        scores.append(2.0 * (p & t).sum().item() / (denom + eps))

    return float(np.mean(scores)) if scores else 1.0


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not mask_a.any() or not mask_b.any():
        return np.array([], dtype=np.float64)

    structure = np.ones((3, 3, 3), dtype=bool)
    surface_a = mask_a ^ binary_erosion(mask_a, structure=structure, border_value=0)
    surface_b = mask_b ^ binary_erosion(mask_b, structure=structure, border_value=0)

    dist_to_b = distance_transform_edt(~surface_b, sampling=spacing)
    dist_to_a = distance_transform_edt(~surface_a, sampling=spacing)

    return np.concatenate([
        dist_to_b[surface_a],
        dist_to_a[surface_b],
    ])


def hd95(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    distances = _surface_distances(pred, target, spacing)
    return float(np.percentile(distances, 95))


def multiclass_hd95(pred: np.ndarray, target: np.ndarray, num_classes: int, spacing=(1.0, 1.0, 1.0), include_background: bool = False) -> float:
    start = 0 if include_background else 1
    values = []

    for c in range(start, num_classes):
        values.append(hd95(pred == c, target == c, spacing))

    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")
