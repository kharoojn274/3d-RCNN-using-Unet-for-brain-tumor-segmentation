from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("t1", "t1ce", "t2", "flair")
LABEL_NAME = "seg"
LABELS = {"background": 0, "necrosis": 1, "edema": 2, "enhancing_tumor": 4}


def discover_subjects(root: str | Path) -> List[Path]:
    """Discover BraTS 2020 training subjects.

    Expected structure:
      BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
        BraTS20_Training_001/
          BraTS20_Training_001_t1.nii.gz
          BraTS20_Training_001_t1ce.nii.gz
          BraTS20_Training_001_t2.nii.gz
          BraTS20_Training_001_flair.nii.gz
          BraTS20_Training_001_seg.nii.gz
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"BraTS 2020 root does not exist: {root}")

    subjects = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        required = [
            subject_dir / f"{subject_id}_{mod}.nii.gz"
            for mod in MODALITIES
        ]
        seg = subject_dir / f"{subject_id}_{LABEL_NAME}.nii.gz"

        if all(p.is_file() for p in required) and seg.is_file():
            subjects.append(subject_dir)

    if not subjects:
        raise RuntimeError(
            f"No valid BraTS 2020 subjects found under {root}. "
            "Expected <ID>/<ID>_t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, "
            "_flair.nii.gz and _seg.nii.gz."
        )
    return subjects


def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data


def validate_subject(subject_dir: str | Path) -> Dict:
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    paths = {
        mod: subject_dir / f"{subject_id}_{mod}.nii.gz"
        for mod in MODALITIES
    }
    seg_path = subject_dir / f"{subject_id}_{LABEL_NAME}.nii.gz"

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if not seg_path.is_file():
        missing.append(str(seg_path))
    if missing:
        raise FileNotFoundError(
            "Missing BraTS 2020 files:\n" + "\n".join(missing)
        )

    reference_img, reference = load_nifti(paths["t1"])
    report = {
        "subject_id": subject_id,
        "shape": tuple(reference.shape),
        "affine": reference_img.affine,
        "spacing": tuple(reference_img.header.get_zooms()[:3]),
        "modalities": {},
        "label_values": None,
    }

    for mod, path in paths.items():
        img, data = load_nifti(path)
        if data.shape != reference.shape:
            raise ValueError(
                f"{subject_id}: {mod} shape {data.shape} != reference {reference.shape}"
            )
        if not np.allclose(img.affine, reference_img.affine, atol=1e-4):
            raise ValueError(f"{subject_id}: {mod} affine does not match t1")

        report["modalities"][mod] = {
            "shape": tuple(data.shape),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
        }

    seg_img, seg = load_nifti(seg_path)
    if seg.shape != reference.shape:
        raise ValueError(
            f"{subject_id}: segmentation shape {seg.shape} != reference {reference.shape}"
        )
    if not np.allclose(seg_img.affine, reference_img.affine, atol=1e-4):
        raise ValueError(f"{subject_id}: segmentation affine mismatch")

    unique = np.unique(seg.astype(np.int16))
    allowed = np.array([0, 1, 2, 4])
    if not np.all(np.isin(unique, allowed)):
        raise ValueError(
            f"{subject_id}: unexpected segmentation labels {unique.tolist()}"
        )
    report["label_values"] = unique.tolist()
    return report


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Map BraTS 2020 labels 0,1,2,4 -> contiguous 0,1,2,3."""
    out = np.zeros(seg.shape, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def zscore_nonzero(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return volume.astype(np.float32)

    values = volume[mask].astype(np.float32)
    mean = values.mean()
    std = values.std()
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / max(float(std), eps)
    return out


class BraTS2020Dataset(Dataset):
    """BraTS 2020 multimodal training dataset.

    Returns image as [4, H, W, D] and mask as [H, W, D].
    """

    def __init__(
        self,
        root: str | Path,
        subjects: Optional[Sequence[Path]] = None,
        transform=None,
        normalize: bool = True,
    ):
        self.root = Path(root)
        self.subjects = list(subjects) if subjects is not None else discover_subjects(self.root)
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int):
        subject_dir = Path(self.subjects[index])
        subject_id = subject_dir.name

        arrays = []
        reference_img = None

        for mod in MODALITIES:
            img, data = load_nifti(subject_dir / f"{subject_id}_{mod}.nii.gz")
            if reference_img is None:
                reference_img = img
            data = data.astype(np.float32)
            if self.normalize:
                data = zscore_nonzero(data)
            arrays.append(data)

        _, seg = load_nifti(subject_dir / f"{subject_id}_{LABEL_NAME}.nii.gz")
        seg = remap_labels(seg)

        sample = {
            "image": torch.from_numpy(np.stack(arrays, axis=0).astype(np.float32)),
            "mask": torch.from_numpy(seg),
            "subject_id": subject_id,
            "affine": reference_img.affine.copy(),
            "header": reference_img.header.copy(),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def validate_dataset(root: str | Path, limit: Optional[int] = None) -> List[Dict]:
    subjects = discover_subjects(root)
    if limit is not None:
        subjects = subjects[:limit]
    return [validate_subject(subject) for subject in subjects]
