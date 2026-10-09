from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint atomically to avoid half-written files on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    best_mean_dice: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    payload = {
        "format_version": 1,
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(completed_epoch) + 1,
        "best_mean_dice": float(best_mean_dice),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "config": config,
        "rng_state": _rng_state(),
    }
    atomic_torch_save(payload, path)


def load_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore all supplied training objects and return metadata/history."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # This file is a user-created training checkpoint, not an untrusted model file.
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        payload = torch.load(checkpoint_path, map_location=map_location)

    if "model_state_dict" not in payload:
        raise KeyError("Checkpoint is missing model_state_dict; cannot resume safely.")

    model.load_state_dict(payload["model_state_dict"], strict=True)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(payload.get("rng_state"))

    return {
        "completed_epoch": int(payload.get("completed_epoch", 0)),
        "next_epoch": int(payload.get("next_epoch", payload.get("completed_epoch", 0) + 1)),
        "best_mean_dice": float(payload.get("best_mean_dice", float("-inf"))),
        "history": list(payload.get("history", [])),
        "config": dict(payload.get("config", {})),
    }


def save_last_and_best(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    mean_dice: float,
    best_mean_dice: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> float:
    """Always save last.pt; update best.pt only when mean Dice improves."""
    checkpoint_dir = Path(checkpoint_dir)
    shared = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=completed_epoch,
        history=history,
        config=config,
    )
    save_training_state(
        checkpoint_dir / "last.pt",
        best_mean_dice=max(best_mean_dice, mean_dice),
        **shared,
    )
    if mean_dice > best_mean_dice:
        save_training_state(
            checkpoint_dir / "best.pt",
            best_mean_dice=mean_dice,
            **shared,
        )
        return float(mean_dice)
    return float(best_mean_dice)
