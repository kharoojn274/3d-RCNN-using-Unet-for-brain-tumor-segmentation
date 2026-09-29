from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    """Configuration for lightweight 3D BraTS training augmentation."""

    enabled: bool = True
    flip_probability: float = 0.5
    rotation_probability: float = 0.25
    max_rotation_90_steps: int = 1
    noise_probability: float = 0.15
    noise_std: float = 0.01
    intensity_probability: float = 0.15
    intensity_scale_range: Tuple[float, float] = (0.9, 1.1)
    intensity_shift_range: Tuple[float, float] = (-0.1, 0.1)


class BraTSAugment:
    """Joint augmentation for MRI image and segmentation mask.

    Expected image shape: [C, D, H, W]
    Expected mask shape:  [D, H, W]

    Spatial transforms are applied identically to image and mask.
    Intensity transforms affect MRI channels only.
    """

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
    ):
        self.config = config or AugmentationConfig()

    @staticmethod
    def _flip(
        image: torch.Tensor,
        mask: Optional[torch.Tensor],
        axis: int,
    ):
        image = torch.flip(image, dims=(axis,))
        if mask is not None:
            mask = torch.flip(mask, dims=(axis - 1,))
        return image, mask

    @staticmethod
    def _rotate_90(
        image: torch.Tensor,
        mask: Optional[torch.Tensor],
        axis_pair: Tuple[int, int],
        k: int,
    ):
        image = torch.rot90(image, k=k, dims=axis_pair)

        if mask is not None:
            mask_axes = (axis_pair[0] - 1, axis_pair[1] - 1)
            mask = torch.rot90(mask, k=k, dims=mask_axes)

        return image, mask

    def _random_spatial(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # image dimensions: C,D,H,W -> spatial axes 1,2,3
        if torch.rand(()) < self.config.flip_probability:
            image, mask = self._flip(image, mask, axis=1)

        if torch.rand(()) < self.config.flip_probability:
            image, mask = self._flip(image, mask, axis=2)

        if torch.rand(()) < self.config.flip_probability:
            image, mask = self._flip(image, mask, axis=3)

        if (
            torch.rand(())
            < self.config.rotation_probability
        ):
            max_steps = max(
                1,
                min(self.config.max_rotation_90_steps, 3),
            )
            k = int(
                torch.randint(
                    1,
                    max_steps + 1,
                    (1,),
                ).item()
            )

            # Rotate within the axial plane.
            image, mask = self._rotate_90(
                image,
                mask,
                axis_pair=(2, 3),
                k=k,
            )

        return image, mask

    def _random_intensity(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if torch.rand(()) >= self.config.intensity_probability:
            return image

        scale_low, scale_high = (
            self.config.intensity_scale_range
        )
        shift_low, shift_high = (
            self.config.intensity_shift_range
        )

        scale = (
            torch.empty((), device=image.device)
            .uniform_(scale_low, scale_high)
        )

        shift = (
            torch.empty((), device=image.device)
            .uniform_(shift_low, shift_high)
        )

        # Preserve zero-valued background.
        foreground = image != 0
        image = image.clone()
        image[foreground] = (
            image[foreground] * scale + shift
        )

        return image

    def _random_noise(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if torch.rand(()) >= self.config.noise_probability:
            return image

        noise = torch.randn_like(image) * self.config.noise_std

        # Do not inject noise into pure background.
        foreground = image != 0

        output = image.clone()
        output[foreground] += noise[foreground]

        return output

    def __call__(
        self,
        sample: Dict,
    ) -> Dict:
        if not self.config.enabled:
            return sample

        image = sample["image"]
        mask = sample.get("mask")

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)

        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)

        image = image.float()

        image, mask = self._random_spatial(
            image,
            mask,
        )

        image = self._random_intensity(image)
        image = self._random_noise(image)

        output = dict(sample)
        output["image"] = image
        output["mask"] = mask

        return output


def build_train_augmentation() -> BraTSAugment:
    """Return the default training augmentation pipeline."""
    return BraTSAugment(AugmentationConfig())


def build_validation_augmentation() -> BraTSAugment:
    """Validation should remain deterministic."""
    return BraTSAugment(
        AugmentationConfig(enabled=False)
    )
