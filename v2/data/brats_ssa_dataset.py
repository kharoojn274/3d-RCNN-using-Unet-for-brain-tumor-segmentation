"""Dataset loader for the Kaggle BraTS-SSA 2025 NIfTI layout.

Expected root: BraTS2025-SSA-TrainingData/
Each subject folder contains:
  <subject>-t1n.nii, <subject>-t1c.nii,
  <subject>-t2w.nii, <subject>-t2f.nii, <subject>-seg.nii
The loader also accepts .nii.gz files.

Tensors preserve the NIfTI voxel-axis order. Image shape is [4, X, Y, Z]
(which Conv3d can consume as [C, D, H, W]); mask shape is [X, Y, Z].
All image and mask arrays for a subject must share shape and affine.
"""

from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
ALLOWED_LABELS = {0, 1, 2, 3}


def _find_exact_suffix(folder: Path, subject_id: str, suffix: str) -> Optional[Path]:
    matches = []
    for extension in (".nii", ".nii.gz"):
        candidate = folder / f"{subject_id}-{suffix}{extension}"
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous files for {subject_id}-{suffix} in {folder}: {matches}"
        )
    return matches[0] if matches else None


def discover_subjects(root_dir: str, require_seg: bool = True) -> List[Dict[str, Path]]:
    """Discover complete subjects under a root containing subject subfolders."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    subjects = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_id = folder.name
        case: Dict[str, Path] = {"case_id": subject_id, "directory": folder}
        missing = []
        for modality in MODALITIES:
            path = _find_exact_suffix(folder, subject_id, modality)
            if path is None:
                missing.append(f"{subject_id}-{modality}.nii[.gz]")
            else:
                case[modality] = path

        seg = _find_exact_suffix(folder, subject_id, "seg")
        if require_seg and seg is None:
            missing.append(f"{subject_id}-seg.nii[.gz]")
        elif seg is not None:
            case["seg"] = seg

        if missing:
            raise FileNotFoundError(f"{subject_id}: missing required files: {missing}")
        subjects.append(case)

    if not subjects:
        raise RuntimeError(f"No subject folders found under {root}")
    return subjects


class BraTSSSADataset(Dataset):
    """BraTS-SSA 2025 dataset returning image, mask, case metadata."""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        load_segmentation: bool = True,
        normalize: bool = True,
        strict: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.load_segmentation = load_segmentation
        self.normalize = normalize
        self.strict = strict
        self.cases = discover_subjects(
            str(self.root_dir), require_seg=load_segmentation
        )

    @staticmethod
    def _load(path: Path):
        image = nib.load(str(path))
        return image.get_fdata(dtype=np.float32), image

    @staticmethod
    def _normalize_volume(volume: np.ndarray) -> np.ndarray:
        nonzero = volume != 0
        if not np.any(nonzero):
            return volume.astype(np.float32, copy=False)
        values = volume[nonzero]
        std = float(values.std())
        out = np.zeros_like(volume, dtype=np.float32)
        if std < 1e-8:
            out[nonzero] = values - float(values.mean())
        else:
            out[nonzero] = (values - float(values.mean())) / std
        return out

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]
        volumes = []
        reference = None

        for modality in MODALITIES:
            volume, nifti = self._load(case[modality])
            if reference is None:
                reference = nifti
            elif self.strict:
                if volume.shape != reference.shape:
                    raise ValueError(
                        f"{case['case_id']} {modality} shape {volume.shape} "
                        f"does not match {reference.shape}"
                    )
                if not np.allclose(nifti.affine, reference.affine, atol=1e-4):
                    raise ValueError(
                        f"{case['case_id']} {modality} affine differs from "
                        f"{MODALITIES[0]}"
                    )
            if self.normalize:
                volume = self._normalize_volume(volume)
            volumes.append(volume.astype(np.float32, copy=False))

        image = np.stack(volumes, axis=0)
        mask = None

        if self.load_segmentation:
            seg_path = case.get("seg")
            if seg_path is None:
                raise FileNotFoundError(f"No segmentation for {case['case_id']}")
            raw_mask, seg_image = self._load(seg_path)
            if raw_mask.shape != image.shape[1:]:
                raise ValueError(
                    f"{case['case_id']} mask shape {raw_mask.shape} "
                    f"does not match image shape {image.shape[1:]}"
                )
            if self.strict and not np.allclose(
                seg_image.affine, reference.affine, atol=1e-4
            ):
                raise ValueError(f"{case['case_id']} mask affine differs from image")
            rounded = np.rint(raw_mask)
            if not np.allclose(raw_mask, rounded, atol=1e-4):
                raise ValueError(f"{case['case_id']} mask contains non-integer labels")
            labels = set(np.unique(rounded).astype(np.int64).tolist())
            unexpected = labels - ALLOWED_LABELS
            if unexpected:
                raise ValueError(
                    f"{case['case_id']} has labels {sorted(unexpected)}; "
                    f"expected a subset of {sorted(ALLOWED_LABELS)}"
                )
            mask = rounded.astype(np.int64, copy=False)

        sample = {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": None if mask is None else torch.from_numpy(mask.copy()).long(),
            "case_id": case["case_id"],
            "affine": reference.affine.copy(),
            "header": reference.header.copy(),
            "spacing_mm": tuple(float(x) for x in reference.header.get_zooms()[:3]),
        }
        return self.transform(sample) if self.transform is not None else sample
