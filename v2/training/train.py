from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.rcnn_unet3d import build_rcnn_unet3d
from .losses import DiceCrossEntropyLoss
from .metrics import dice_score


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(image)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.detach().item()

    return running / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        logits = model(image)
        total_loss += criterion(logits, target).item()
        total_dice += dice_score(logits, target, include_background=False)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_dice": best_dice,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    raise RuntimeError(
        "Connect the BraTS-GLI DataLoader here before training. "
        "The model/loss/validation pipeline is ready, but dataset-specific "
        "batch construction must match the actual downloaded 2023 files."
    )


if __name__ == "__main__":
    main()
