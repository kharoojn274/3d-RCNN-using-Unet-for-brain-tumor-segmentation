# V2 cleanup plan — BraTS-SSA 2025

This branch is a safe refactor track. The original `v2-research` branch is preserved.

## Dataset decision
- Current dataset: Kaggle BraTS-SSA 2025, with 60 training subject folders confirmed by the user.
- The user's file audit found all five expected files in every training subject: T1n, T1c, T2w, T2f, and segmentation.
- Files are `.nii`; loaders must support both `.nii` and `.nii.gz`.
- Do not mix BraTS 2020 and BraTS-SSA cases in one run unless cohort provenance and label conversion are explicit.
- The sample case has shape 240×240×155, spacing 1×1×1 mm, and segmentation labels 0, 1, 2, 3. The complete cohort still needs the new audit script run.
- The BraTS-SSA 2025 label meanings must be confirmed from dataset/challenge documentation before interpreting WT/TC/ET results in a paper. The expected standard BraTS-Africa mapping is 0 background, 1 non-enhancing tumor core, 2 surrounding non-enhancing FLAIR hyperintensity/edema, 3 enhancing tumor.

## Canonical pipeline
1. Environment and config
2. Dataset discovery and full integrity audit
3. Fixed patient-level train/validation split with leakage check
4. Preprocessing and patch sampling (one documented implementation)
5. Model, loss, optimizer, scheduler, AMP
6. One-batch forward/backward smoke test
7. One-patient validation smoke test
8. One centralized WT/TC/ET metric implementation, after label meanings are confirmed
9. One training loop and one history CSV
10. Atomic last/best checkpointing with full resume state
11. Ten resumable training segments: epochs 1–5, 6–10, …, 46–50
12. Final evaluation and experiment report

## Acceptance gates before expensive GPU training
- [ ] Full BraTS-SSA audit passes on all 60 training subjects.
- [ ] Train and validation subject IDs do not overlap.
- [ ] Label meanings are confirmed from challenge/dataset documentation.
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
- Do not call validation challenge cases a labeled validation set if their segmentation masks are not provided. Use a held-out subset of labeled training cases for model selection, then evaluate separately on any official labeled test set if available.
