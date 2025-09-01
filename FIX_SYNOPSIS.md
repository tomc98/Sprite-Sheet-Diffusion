# Pose Guider Integration Fix — Synopsis

## Overview
This change resolves a runtime API mismatch in the pose-conditioning path that caused the diffusion pipeline to fail and fall back to a simple overlay. We made the pipeline tolerant to both PoseGuider call signatures and adapted the original `PoseGuiderOrg` to output multi‑resolution features required by `UNet3D`.

## Why
- Pipeline called `pose_guider(pose_cond, ref_pose)` while `PoseGuiderOrg.forward` accepts only one tensor → `TypeError` at inference.
- `UNet3D` expects pose features as a list of tensors with specific channel sizes per block (320, 320, 640, 1280, 1280). `PoseGuiderOrg` produces a single 320‑channel tensor. A lightweight adapter is needed.

## What Changed
- Added a safe, dual‑signature call site in the pipeline:
  - Tries two‑arg call (`pose_cond, ref_pose`) first; on `TypeError`, falls back to one‑arg (`pose_cond`).
  - Handles classifier‑free guidance (CFG) for both list and tensor outputs.
- Wrapped `PoseGuiderOrg` with a small adapter that produces the expected multi‑resolution features for `UNet3D` without retraining.

## Code References
- Dual‑signature PoseGuider call and CFG handling:
  - Sprite-Sheet-Diffusion/ModelTraining/pipelines/pipeline_pose2img.py:303
  - Sprite-Sheet-Diffusion/ModelTraining/pipelines/pipeline_pose2img.py:308
- Import the adapter in the real webapp:
  - Sprite-Sheet-Diffusion/webapp/app_real_diffusion_pipeline.py:31
- Load `PoseGuiderOrg`, then wrap with `SingleTensorPoseGuider` using correct per‑block channels:
  - Sprite-Sheet-Diffusion/webapp/app_real_diffusion_pipeline.py:120
  - Sprite-Sheet-Diffusion/webapp/app_real_diffusion_pipeline.py:130

## Technical Notes
- `SingleTensorPoseGuider` adapts the single 320‑ch output into a list sized for `UNet3D` blocks: `(320, 320, 640, 1280, 1280)` with appropriate spatial resolutions. This keeps compatibility with `pose_guider.pth` weights.
- The pipeline’s try/except keeps compatibility with both the older 2‑arg `PoseGuider` and the 1‑arg `PoseGuiderOrg` + adapter.

## Validation
- Start the real diffusion webapp and trigger a "walk" generation.
- Expected: 8 distinct frames with realistic pose variation; no overlay fallback.
- Output files to inspect:
  - Results: `webapp/static/results/real_diffusion_sprite_walk_*.png`
  - Logs (if enabled): `webapp/webapp_fixed.log`

## Next Steps (Optional)
- Apply the same dual‑signature pattern to the video pipelines (`pipeline_pose2vid*.py`).
- Once fully standardized on the adapter, the try/except can be removed and the one‑arg call used consistently.
