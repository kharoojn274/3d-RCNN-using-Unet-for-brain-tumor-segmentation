from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from v2.data.brats_gli_dataset import (
    BraTSGLI2023Dataset,
    discover_subjects,
    validate_subject,
)


def verify_split(train_root, val_root):
    train = discover_subjects(train_root, require_seg=True)
    val = discover_subjects(val_root, require_seg=False)

    train_ids = {p.name for p in train}
    val_ids = {p.name for p in val}
    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            f"Train/validation leakage detected: {sorted(overlap)[:10]}"
        )

    report = {
        "train_count": len(train),
        "validation_count": len(val),
        "overlap_count": len(overlap),
        "train_examples": sorted(train_ids)[:5],
        "validation_examples": sorted(val_ids)[:5],
    }

    print(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def run_one_patient(dataset_root, model, device, require_seg=True):
    subjects = discover_subjects(dataset_root, require_seg=require_seg)
    subject = random.choice(subjects)

    report = validate_subject(subject, require_seg=require_seg)
    dataset = BraTSGLI2023Dataset(
        dataset_root,
        subjects=[subject],
        require_seg=require_seg,
        normalize=True,
    )

    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    output = model(image)

    if isinstance(output, (tuple, list)):
        output = output[0]

    prediction = torch.argmax(output, dim=1)

    result = {
        "subject_id": sample["subject_id"],
        "input_shape": list(image.shape),
        "output_shape": list(output.shape),
        "prediction_shape": list(prediction.shape),
        "prediction_labels": torch.unique(prediction).detach().cpu().tolist(),
        "reference_shape": list(report["shape"]),
        "spacing": list(report["spacing"]),
    }

    if "mask" in sample:
        result["ground_truth_shape"] = list(sample["mask"].shape)
        result["ground_truth_labels"] = (
            torch.unique(sample["mask"]).cpu().tolist()
        )

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify BraTS-GLI split and run one real patient."
    )
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--smoke-test-root", required=True)
    parser.add_argument(
        "--mode",
        choices=("split", "patient"),
        default="split",
    )
    args = parser.parse_args()

    if args.mode == "split":
        verify_split(args.train_root, args.val_root)
        return

    # Import only when the patient smoke test is requested so the script
    # can be used for split verification without the model dependencies.
    from v2.models.rcnn_unet_v2 import RCNNUNetV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNNUNetV2(
        in_channels=4,
        num_classes=4,
    )

    run_one_patient(
        args.smoke_test_root,
        model=model,
        device=device,
        require_seg=True,
    )


if __name__ == "__main__":
    main()
