# V2 cleanup plan — BraTS-GLI 2023

This branch is a safe refactor track. The existing `v2-research` branch is preserved.

## Dataset decision
- Primary dataset: BraTS 2023 Adult Glioma (BraTS-GLI), after the user downloads it from the official Synapse project and accepts its terms.
- Do not mix BraTS 2020 and BraTS-GLI files in a single training run unless the label mapping and cohort provenance are explicit.
- Validate every subject before training: required modalities, segmentation presence, spatial shape, affine, voxel spacing, and label values.
- BraTS 2020 enhancing-tumor label `4` and BraTS-GLI label `3` must never be treated as interchangeable without an explicit mapping.

## Canonical pipeline
1. Environment and config
2. Dataset discovery and integrity audit
3. Patient-level train/validation split with leakage check
4. Preprocessing and patch sampling (one documented implementation)
5. Model, loss, optimizer, scheduler, AMP
6. One-batch forward/backward smoke test
7. One-patient validation smoke test
8. One centralized WT/TC/ET metric implementation
9. One training loop and one history CSV
10. Atomic last/best checkpointing with full resume state
11. Ten resumable training segments: epochs 1–5, 6–10, …, 46–50
12. Final evaluation and experiment report

## Acceptance gates before expensive GPU training
- [ ] Dataset audit passes on all training subjects.
- [ ] Train and validation subject IDs do not overlap.
- [ ] Label IDs are confirmed from the actual downloaded data.
- [ ] Model output has 4 logits and target masks contain class IDs 0–3.
- [ ] Input/target spatial shapes match the model output.
- [ ] One real batch completes forward, loss, backward, optimizer step.
- [ ] One real validation subject yields WT/TC/ET metrics.
- [ ] Resume test starts at the next epoch and restores optimizer/scheduler/scaler/history.
- [ ] A checkpoint is copied to durable storage outside the ephemeral runtime.

## Important scientific rules
- Report validation metrics only from the real model; keep synthetic smoke-test outputs separate.
- Compute HD95 using voxel spacing from the NIfTI header, not a hard-coded 1 mm spacing.
- Keep a fixed held-out patient split and record random seed, code commit, dataset version, and config.
- BraTS-GLI's 2023 label scheme is 0 background, 1 NCR/NETC, 2 edema, 3 enhancing tumor (ET). Derive regions from these labels:
  - WT = labels 1, 2, or 3
  - TC = labels 1 or 3
  - ET = label 3
