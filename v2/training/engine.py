from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 12.0
    amp: bool = True
    accumulation_steps: int = 1
    checkpoint_dir: str = "checkpoints"
    monitor: str = "dice"


def _move_batch(batch, device):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return image, mask


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, config):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_amp = config.amp and device.type == "cuda"

    for step, batch in enumerate(loader):
        image, mask = _move_batch(batch, device)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            loss = criterion(logits, mask)
            loss = loss / config.accumulation_steps

        scaler.scale(loss).backward()

        should_step = (
            (step + 1) % config.accumulation_steps == 0
            or (step + 1) == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach().item() * config.accumulation_steps

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, dice_fn, hd95_fn, device, config):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_hd95 = 0.0
    count = 0

    for batch in loader:
        image, mask = _move_batch(batch, device)

        with autocast(
            device_type=device.type,
            enabled=config.amp and device.type == "cuda",
        ):
            logits = model(image)
            loss = criterion(logits, mask)

        total_loss += loss.item()

        pred = torch.argmax(logits, dim=1)

        total_dice += dice_fn(logits, mask)

        # HD95 is calculated on CPU because the reference implementation
        # operates on NumPy/scipy arrays.
        total_hd95 += hd95_fn(
            pred.detach().cpu().numpy(),
            mask.detach().cpu().numpy(),
        )

        count += 1

    count = max(count, 1)

    return {
        "loss": total_loss / count,
        "dice": total_dice / count,
        "hd95": total_hd95 / count,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": vars(config),
        },
        path,
    )


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    dice_fn,
    hd95_fn,
    device,
    config,
):
    scaler = GradScaler(
        device="cuda",
        enabled=config.amp and device.type == "cuda",
    )

    best_dice = float("-inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            config,
        )

        metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            dice_fn,
            hd95_fn,
            device,
            config,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"dice={metrics['dice']:.4f} | "
            f"HD95={metrics['hd95']:.2f}"
        )

        save_checkpoint(
            Path(config.checkpoint_dir) / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            config,
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]

            save_checkpoint(
                Path(config.checkpoint_dir) / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                config,
            )

    return history
