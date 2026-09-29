# 3D RCNN-U-Net Research Framework — V2

This branch contains the V2 rebuild of the original brain-tumor segmentation project.

## Scope
- Tasks 1–3: 3D segmentation research pipeline built around a properly modular 3D RCNN-U-Net.
- Task 4: MRI inpainting pipeline (separate architecture).
- Task 5: histopathology classification pipeline (separate architecture).

## Principles
1. Keep the original V1 implementation unchanged.
2. Reuse validated ideas/components from V1 only after inspection.
3. Keep task-specific datasets and rules isolated.
4. Preserve NIfTI spatial metadata during inference.
5. Add reproducible configs, metrics, experiments, and Docker support.

## Status
V2 scaffold started. Model and data modules will be added incrementally after dataset/task specifications are locked.
