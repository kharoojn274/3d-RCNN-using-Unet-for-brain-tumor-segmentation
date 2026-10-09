"""Integrity audit for BraTS-SSA 2025 folders containing .nii or .nii.gz files.

Example:
python -m v2.data.audit_brats_ssa --root /kaggle/input/.../BraTS2025-SSA-TrainingData
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from v2.data.brats_ssa_dataset import ALLOWED_LABELS, MODALITIES, discover_subjects


def audit_subject(case: dict) -> dict:
    case_id = case["case_id"]
    images = {}
    for name in MODALITIES:
        images[name] = nib.load(str(case[name]))
    images["seg"] = nib.load(str(case["seg"]))

    reference = images[MODALITIES[0]]
    shape = reference.shape
    affine = reference.affine
    spacing = tuple(float(v) for v in reference.header.get_zooms()[:3])

    for name, image in images.items():
        if image.shape != shape:
            raise ValueError(f"{name} shape {image.shape} != {shape}")
        if not np.allclose(image.affine, affine, atol=1e-4):
            raise ValueError(f"{name} affine differs from {MODALITIES[0]}")

    seg = np.asarray(images["seg"].dataobj)
    if not np.isfinite(seg).all():
        raise ValueError("segmentation contains NaN/Inf")
    rounded = np.rint(seg)
    if not np.allclose(seg, rounded, atol=1e-4):
        raise ValueError("segmentation contains non-integer values")
    labels, counts = np.unique(rounded.astype(np.int16), return_counts=True)
    label_counts = {str(int(k)): int(v) for k, v in zip(labels, counts)}
    unexpected = set(labels.astype(int).tolist()) - ALLOWED_LABELS
    if unexpected:
        raise ValueError(f"unexpected labels: {sorted(unexpected)}")

    modality_stats = {}
    for name in MODALITIES:
        arr = np.asarray(images[name].dataobj, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")
        modality_stats[name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
        }

    return {
        "case_id": case_id,
        "shape": list(shape),
        "spacing_mm": list(spacing),
        "labels": [int(v) for v in labels],
        "label_counts": label_counts,
        "modalities": modality_stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Training root with subject folders")
    parser.add_argument("--limit", type=int, default=0, help="0 means all subjects")
    parser.add_argument("--output", default="reports/brats_ssa_audit.json")
    args = parser.parse_args()

    cases = discover_subjects(args.root, require_seg=True)
    if args.limit > 0:
        cases = cases[:args.limit]

    reports, failures = [], []
    for case in cases:
        try:
            reports.append(audit_subject(case))
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": str(exc)})

    summary = {
        "dataset_root": str(Path(args.root).resolve()),
        "subjects_checked": len(cases),
        "subjects_passed": len(reports),
        "subjects_failed": len(failures),
        "failures": failures,
        "subjects": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"}, indent=2))
    print(f"Full report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
