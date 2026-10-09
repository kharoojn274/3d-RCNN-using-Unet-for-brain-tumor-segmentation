from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from v2.data.brats_gli_dataset import MODALITIES, discover_subjects


ALLOWED_LABELS = {0, 1, 2, 3}


def audit_subject(subject_dir: Path) -> dict:
    subject_id = subject_dir.name
    paths = {
        modality: subject_dir / f"{subject_id}-{modality}.nii.gz"
        for modality in MODALITIES
    }
    paths["seg"] = subject_dir / f"{subject_id}-seg.nii.gz"

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{subject_id}: missing files: {missing}")

    images = {name: nib.load(str(path)) for name, path in paths.items()}
    reference = images[MODALITIES[0]]
    reference_shape = reference.shape
    reference_affine = reference.affine
    spacing = tuple(float(x) for x in reference.header.get_zooms()[:3])

    for name, image in images.items():
        if image.shape != reference_shape:
            raise ValueError(
                f"{subject_id}: {name} shape {image.shape} != {reference_shape}"
            )
        if not np.allclose(image.affine, reference_affine, atol=1e-4):
            raise ValueError(f"{subject_id}: {name} affine differs from {MODALITIES[0]}")

    seg = np.asarray(images["seg"].dataobj)
    labels = {int(x) for x in np.unique(seg)}
    unexpected = labels - ALLOWED_LABELS
    if unexpected:
        raise ValueError(
            f"{subject_id}: unexpected BraTS-GLI labels {sorted(unexpected)}; "
            f"expected a subset of {sorted(ALLOWED_LABELS)}"
        )

    modality_stats = {}
    for name in MODALITIES:
        arr = np.asarray(images[name].dataobj, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{subject_id}: {name} contains NaN or Inf")
        modality_stats[name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
        }

    return {
        "subject_id": subject_id,
        "shape": list(reference_shape),
        "spacing_mm": list(spacing),
        "labels": sorted(labels),
        "modalities": modality_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrity audit for extracted BraTS-GLI 2023 training data."
    )
    parser.add_argument("--root", required=True, help="Root containing subject folders")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit; 0 audits all")
    parser.add_argument("--output", default="reports/brats_gli_audit.json")
    args = parser.parse_args()

    subjects = discover_subjects(args.root, require_seg=True)
    if args.limit > 0:
        subjects = subjects[: args.limit]

    reports = []
    failures = []
    for subject in subjects:
        try:
            reports.append(audit_subject(subject))
        except Exception as exc:
            failures.append({"subject_id": subject.name, "error": str(exc)})

    summary = {
        "dataset_root": str(Path(args.root).resolve()),
        "subjects_checked": len(subjects),
        "subjects_passed": len(reports),
        "subjects_failed": len(failures),
        "failures": failures,
        "subjects": reports,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"}, indent=2))
    print(f"Full audit report: {output}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
