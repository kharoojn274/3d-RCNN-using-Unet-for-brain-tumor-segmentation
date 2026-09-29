from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        if not self.include_background and num_classes > 1:
            probs, target_oh = probs[:, 1:], target_oh[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = (probs * target_oh).sum(dims)
        denominator = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class CrossEntropyLoss3D(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target.long())


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(include_background=False)
        self.ce = CrossEntropyLoss3D()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module | None = None, weights: tuple[float, ...] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.base_loss = base_loss or DiceCrossEntropyLoss()
        self.weights = weights

    def forward(self, outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, target)

        if len(outputs) != len(self.weights):
            raise ValueError("Number of outputs must match number of deep-supervision weights.")

        loss = outputs[0].new_tensor(0.0)
        for output, weight in zip(outputs, self.weights):
            if output.shape[2:] != target.shape[1:]:
                output = F.interpolate(output, size=target.shape[1:], mode="trilinear", align_corners=False)
            loss = loss + weight * self.base_loss(output, target)

        return loss
