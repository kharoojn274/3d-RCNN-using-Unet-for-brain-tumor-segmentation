from __future__ import annotations

import torch


REGIONS = ("WT", "TC", "ET")


def class_ids_to_regions(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert BraTS-GLI class IDs (0,1,2,3) into nested WT/TC/ET masks.

    Label mapping: 0=background, 1=NCR/NETC, 2=edema, 3=ET.
    Accepts [D,H,W] or [B,D,H,W] integer label tensors.
    """
    if labels.ndim not in (3, 4):
        raise ValueError(f"Expected [D,H,W] or [B,D,H,W], got {tuple(labels.shape)}")
    return {
        "WT": (labels == 1) | (labels == 2) | (labels == 3),
        "TC": (labels == 1) | (labels == 3),
        "ET": labels == 3,
    }


def logits_to_regions(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert [B,4,D,H,W] multiclass logits to WT/TC/ET masks."""
    if logits.ndim != 5 or logits.shape[1] != 4:
        raise ValueError(f"Expected logits [B,4,D,H,W], got {tuple(logits.shape)}")
    return class_ids_to_regions(torch.argmax(logits, dim=1))


def dice_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-case binary Dice; both-empty regions score 1."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    pred = pred.bool()
    target = target.bool()
    if pred.ndim < 2:
        raise ValueError("Expected a batch dimension followed by spatial dimensions")
    dims = tuple(range(1, pred.ndim))
    intersection = (pred & target).sum(dim=dims).float()
    denominator = pred.sum(dim=dims).float() + target.sum(dim=dims).float()
    return torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        (2.0 * intersection + eps) / (denominator + eps),
    )


def region_dice(logits: torch.Tensor, target_labels: torch.Tensor) -> dict[str, float]:
    """Return mean-over-cases Dice for WT, TC and ET.

    logits: [B,4,D,H,W]; target_labels: [B,D,H,W] with class IDs 0..3.
    """
    if target_labels.ndim != 4:
        raise ValueError(
            f"Expected target [B,D,H,W], got {tuple(target_labels.shape)}"
        )
    pred_regions = logits_to_regions(logits)
    target_regions = class_ids_to_regions(target_labels.long())
    result = {}
    for region in REGIONS:
        scores = dice_binary(pred_regions[region], target_regions[region])
        result[region] = float(scores.mean().item())
    result["mean_dice"] = sum(result[r] for r in REGIONS) / len(REGIONS)
    return result
