from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch


@dataclass
class PreprocessConfig:
    """Configuration for BraTS-GLI preprocessing.

    We deliberately do not resample by default. BraTS spatial metadata
    should be preserved until the dataset-specific geometry is verified.
    """

    normalize: bool = True
    clip_percentiles: Optional[Tuple[float, float]] = None
    crop_to_nonzero: bool = False
    label_map: Optional[dict] = None


def load_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI image as float32 data plus its NIfTI object."""
    image = nib.load(path)
    return image.get_fdata(dtype=np.float32), image


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize non-background voxels while keeping background zero."""
    volume = np.asarray(volume, dtype=np.float32)
    mask = volume != 0

    if not np.any(mask):
        return volume.copy()

    values = volume[mask]
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-8:
        return volume.copy()

    output = np.zeros_like(volume, dtype=np.float32)
    output[mask] = (volume[mask] - mean) / std
    return output


def percentile_clip(
    volume: np.ndarray,
    low: float = 0.5,
    high: float = 99.5,
) -> np.ndarray:
    """Clip non-zero intensities to robust percentile limits."""
    volume = np.asarray(volume, dtype=np.float32)
    mask = volume != 0

    if not np.any(mask):
        return volume.copy()

    lo, hi = np.percentile(volume[mask], [low, high])
    output = volume.copy()
    output[mask] = np.clip(output[mask], lo, hi)
    return output


def remap_labels(
    segmentation: np.ndarray,
    label_map: Optional[dict] = None,
) -> np.ndarray:
    """Remap segmentation labels.

    Default keeps BraTS labels unchanged: 0, 1, 2, 4.
    A mapping can later be supplied for a model-specific target encoding.
    """
    segmentation = np.asarray(segmentation)

    if label_map is None:
        return segmentation.astype(np.int16, copy=True)

    output = np.zeros_like(segmentation, dtype=np.int16)
    for source, target in label_map.items():
        output[segmentation == source] = target
    return output


def compute_nonzero_bbox(
    volume: np.ndarray,
) -> Optional[Tuple[slice, slice, slice]]:
    """Return the bounding box of non-zero voxels."""
    coords = np.argwhere(volume != 0)

    if coords.size == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    return tuple(
        slice(int(start), int(end))
        for start, end in zip(mins, maxs)
    )


def crop_to_bbox(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
):
    """Crop an image [C,D,H,W] to its non-zero spatial bounding box.

    Cropping is optional and should only be enabled after geometry
    verification because output reconstruction must preserve spatial metadata.
    """
    spatial = np.any(image != 0, axis=0)
    bbox = compute_nonzero_bbox(spatial)

    if bbox is None:
        return image, mask, None

    cropped_image = image[(slice(None),) + bbox]
    cropped_mask = None if mask is None else mask[bbox]

    return cropped_image, cropped_mask, bbox


def preprocess_modalities(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Preprocess stacked MRI modalities [C,D,H,W]."""
    config = config or PreprocessConfig()

    output = np.asarray(image, dtype=np.float32).copy()

    for channel in range(output.shape[0]):
        if config.clip_percentiles is not None:
            low, high = config.clip_percentiles
            output[channel] = percentile_clip(
                output[channel], low, high
            )

        if config.normalize:
            output[channel] = normalize_nonzero(output[channel])

    return output


def prepare_sample(
    image: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    config: Optional[PreprocessConfig] = None,
):
    """Prepare one loaded BraTS sample for the training pipeline."""
    config = config or PreprocessConfig()

    image = preprocess_modalities(image, config)

    if segmentation is not None:
        segmentation = remap_labels(
            segmentation,
            config.label_map,
        )

    if config.crop_to_nonzero:
        image, segmentation, bbox = crop_to_bbox(
            image,
            segmentation,
        )
    else:
        bbox = None

    return (
        torch.from_numpy(image.copy()).float(),
        None if segmentation is None else torch.from_numpy(
            segmentation.copy()
        ).long(),
        bbox,
    )
