from pathlib import Path
from typing import Dict, List, Tuple
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSGLIDataset(Dataset):
    """BraTS-GLI 2023 NIfTI dataset loader.

    Returns image [4, D, H, W] in modality order:
    T1n, T1c, T2w, T2f.
    Segmentation labels are preserved as 0, 1, 2, 4.
    """

    MODALITIES = ("t1n", "t1c", "t2w", "t2f")

    def __init__(self, root_dir: str, transform=None,
                 load_segmentation: bool = True, normalize: bool = True,
                 strict: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.load_segmentation = load_segmentation
        self.normalize = normalize
        self.strict = strict
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")
        self.cases = self._discover_cases()
        if not self.cases:
            raise RuntimeError(f"No BraTS-GLI cases found in: {self.root_dir}")

    def _discover_cases(self) -> List[Dict[str, Path]]:
        cases = []
        for case_dir in sorted(self.root_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            files = list(case_dir.glob("*.nii.gz"))
            if not files:
                continue
            case = {"case_id": case_dir.name, "directory": case_dir}
            for modality in self.MODALITIES:
                matches = [f for f in files if f"-{modality}.nii.gz" in f.name]
                if len(matches) == 1:
                    case[modality] = matches[0]
                elif self.strict:
                    raise FileNotFoundError(f"Missing or ambiguous {modality} file in {case_dir}")
            if self.load_segmentation:
                matches = [f for f in files if "-seg.nii.gz" in f.name]
                if len(matches) == 1:
                    case["seg"] = matches[0]
                elif self.strict:
                    raise FileNotFoundError(f"Missing or ambiguous segmentation file in {case_dir}")
            cases.append(case)
        return cases

    @staticmethod
    def _load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        image = nib.load(str(path))
        return image.get_fdata(dtype=np.float32), image

    @staticmethod
    def _normalize_volume(volume: np.ndarray) -> np.ndarray:
        mask = volume != 0
        if not np.any(mask):
            return volume.astype(np.float32)
        values = volume[mask]
        std = values.std()
        if std < 1e-8:
            return volume.astype(np.float32)
        out = np.zeros_like(volume, dtype=np.float32)
        out[mask] = (volume[mask] - values.mean()) / std
        return out

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict:
        case = self.cases[index]
        volumes, reference = [], None

        for modality in self.MODALITIES:
            volume, nifti = self._load_nifti(case[modality])
            reference = reference or nifti
            if self.normalize:
                volume = self._normalize_volume(volume)
            volumes.append(volume)

        image = np.stack(volumes, axis=0)
        mask = None

        if self.load_segmentation and "seg" in case:
            mask, _ = self._load_nifti(case["seg"])
            mask = mask.astype(np.int16)
            if mask.shape != image.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for {case['case_id']}: "
                    f"image={image.shape[1:]}, mask={mask.shape}"
                )

        sample = {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": None if mask is None else torch.from_numpy(mask.copy()).long(),
            "case_id": case["case_id"],
            "affine": reference.affine.copy(),
            "header": reference.header.copy(),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
