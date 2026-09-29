from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class DataConfig:
    train_dir: str = "data/BraTS2023_Training"
    val_dir: str = "data/BraTS2023_Validation"
    modalities: tuple = ("t1c", "t1n", "t2f", "t2w")
    target_spacing: tuple = (1.0, 1.0, 1.0)
    patch_size: tuple = (128, 128, 128)
    samples_per_volume: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    in_channels: int = 4
    num_classes: int = 4
    base_channels: int = 16
    recurrent_steps: int = 2
    dropout: float = 0.1


@dataclass
class OptimConfig:
    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class LossConfig:
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    deep_supervision: bool = False


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1
    accumulation_steps: int = 4
    amp: bool = True
    grad_clip: float = 12.0
    seed: int = 42
    checkpoint_dir: str = "checkpoints/v2"
    log_dir: str = "runs/v2"
    save_every: int = 1
    validate_every: int = 1


@dataclass
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()

    def save(self, path: str = "configs/brats2023_v2.json"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = asdict(self)
        payload["data"]["modalities"] = list(payload["data"]["modalities"])
        payload["data"]["target_spacing"] = list(payload["data"]["target_spacing"])
        payload["data"]["patch_size"] = list(payload["data"]["patch_size"])
        payload["model"] = dict(payload["model"])
        payload["optimizer"]["betas"] = list(payload["optimizer"]["betas"])

        path.write_text(json.dumps(payload, indent=2))
        return path


def build_optimizer(model, config: ExperimentConfig):
    import torch

    if config.optimizer.name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )


def build_scheduler(optimizer, config: ExperimentConfig):
    import torch

    total_epochs = config.train.epochs
    warmup_epochs = config.scheduler.warmup_epochs
    min_lr = config.scheduler.min_lr
    base_lr = config.optimizer.learning_rate

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)

        progress = (epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs
        )
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        return float(min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_experiment(config: ExperimentConfig):
    set_seed(config.train.seed)

    Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.log_dir).mkdir(parents=True, exist_ok=True)

    return config
