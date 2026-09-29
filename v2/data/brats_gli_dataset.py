from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
LABELS = {"background": 0, "ncr": 1, "ed": 2, "et": 3}


def discover_subjects(root: str | Path, require_seg: bool = True) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"BraTS root does not exist: {root}")

    subjects = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        required = [
            subject_dir / f"{subject_id}-{mod}.nii.gz"
            for mod in MODALITIES
        ]
        seg = subject_dir / f"{subject_id}-seg.nii.gz"

        if all(p.is_file() for p in required) and (seg.is_file() or not require_seg):
            subjects.append(subject_dir)

    if not subjects:
        raise RuntimeError(
            f"No valid BraTS-GLI subjects found under {root}. "
            "Expected <ID>/<ID>-t1c.nii.gz, -t1n.nii.gz, -t2f.nii.gz, -t2w.nii.gz "
            "and, for training, -seg.nii.gz."
        )
    return subjects


def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return img, data


def validate_subject(subject_dir: str | Path, require_seg: bool = True) -> Dict:
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    paths = {
        mod: subject_dir / f"{subject_id}-{mod}.nii.gz"
        for mod in MODALITIES
    }
    seg_path = subject_dir / f"{subject_id}-seg.nii.gz"

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if require_seg and not seg_path.is_file():
        missing.append(str(seg_path))

    if missing:
        raise FileNotFoundError(
            "Missing BraTS files:\n" + "\n".join(missing)
        )

    reference_img, reference = load_nifti(paths["t1n"])
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
                f"{subject_id}: {mod} shape {data.shape} != "
                f"reference {reference.shape}"
            )
        if not np.allclose(img.affine, reference_img.affine, atol=1e-4):
            raise ValueError(
                f"{subject_id}: {mod} affine does not match t1n"
            )
        report["modalities"][mod] = {
            "shape": tuple(data.shape),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
        }

    if require_seg:
        seg_img, seg = load_nifti(seg_path)
        if seg.shape != reference.shape:
            raise ValueError(
                f"{subject_id}: segmentation shape {seg.shape} != "
                f"reference {reference.shape}"
            )
        if not np.allclose(seg_img.affine, reference_img.affine, atol=1e-4):
            raise ValueError(f"{subject_id}: segmentation affine mismatch")

        unique = np.unique(seg.astype(np.int16))
        allowed = np.array([0, 1, 2, 3])
        if not np.all(np.isin(unique, allowed)):
            raise ValueError(
                f"{subject_id}: unexpected segmentation labels {unique.tolist()}"
            )
        report["label_values"] = unique.tolist()

    return report


def zscore_nonzero(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return volume.astype(np.float32)

    values = volume[mask]
    mean = values.mean()
    std = values.std()
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / max(float(std), eps)
    return out


class BraTSGLI2023Dataset(Dataset):
    """Loads BraTS-GLI 2023 subjects as [C, H, W, D] tensors.

    The official GLI data uses T1c, T1n, T2f and T2w modalities.
    Training cases additionally contain a -seg.nii.gz mask.
    """

    def __init__(
        self,
        root: str | Path,
        subjects: Optional[Sequence[Path]] = None,
        transform=None,
        require_seg: bool = True,
        normalize: bool = True,
    ):
        self.root = Path(root)
        self.subjects = (
            list(subjects)
            if subjects is not None
            else discover_subjects(self.root, require_seg=require_seg)
        )
        self.transform = transform
        self.require_seg = require_seg
        self.normalize = normalize

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int):
        subject_dir = Path(self.subjects[index])
        subject_id = subject_dir.name

        arrays = []
        reference_img = None

        for mod in MODALITIES:
            path = subject_dir / f"{subject_id}-{mod}.nii.gz"
            img, data = load_nifti(path)
            if reference_img is None:
                reference_img = img
            if self.normalize:
                data = zscore_nonzero(data)
            arrays.append(data)

        image = np.stack(arrays, axis=0).astype(np.float32)

        sample = {
            "image": torch.from_numpy(image),
            "subject_id": subject_id,
            "affine": reference_img.affine.copy(),
            "header": reference_img.header.copy(),
        }

        if self.require_seg:
            seg_path = subject_dir / f"{subject_id}-seg.nii.gz"
            _, seg = load_nifti(seg_path)
            seg = seg.astype(np.int64)
            sample["mask"] = torch.from_numpy(seg)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def validate_dataset(root: str | Path, require_seg: bool = True) -> List[Dict]:
    reports = []
    for subject in discover_subjects(root, require_seg=require_seg):
        reports.append(validate_subject(subject, require_seg=require_seg))
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Extracted BraTS-GLI training-data directory")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    subjects = discover_subjects(args.root, require_seg=True)
    for subject in subjects[:args.limit]:
        report = validate_subject(subject)
        print(
            report["subject_id"],
            report["shape"],
            report["spacing"],
            report["label_values"],
        )
