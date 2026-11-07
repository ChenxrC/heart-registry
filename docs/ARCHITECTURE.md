# Code map and call flow

## Main 2-D/3-D registration package

`ctdr_pipeline` is the supported, installable core. The processing stages are
separated so that data linkage, optimization, and evaluation remain auditable:

1. `pptx/` reads physician-marked slide pairs, extracts image coordinates, and
   exports accepted 2-D affine transforms with provenance.
2. `dicom/` scans DICOM metadata, selects CT/localizer/XA series, links the
   annotation bundle to source images, and defines projection geometry.
3. `registration/` builds CT volumes and ROI masks, generates initial poses,
   and runs differentiable DRR pose optimization.
4. `pipeline.py` runs per-case stages with hashes, cache/resume behavior, and
   isolated failure handling.
5. `evaluation/` records similarity, field-of-view, pose-stability, and review
   gates. Missing independent 3-D truth remains missing; it is not inferred
   from the optimizer's own residual.

Call the Python API from `heart_registry` or use the `ctdr` command. The public
facade deliberately requires explicit data, configuration, and output paths.

## Reconstruction and dynamic display

- `ctdr_pipeline/visualization/surface_mesh.py`: CT label/volume to audited,
  smoothed surface mesh variants.
- `tools/xa_cine.py`: XA cine decoding, frame timing, raw stacks, previews, and
  source metadata.
- `tools/run_xa_vessel_segmentation.py`: DeepSA plus a weak Frangi prior and
  frame-level quality summaries. External DeepSA source and model weights are
  required and are not included.
- `tools/xa_temporal_repair.py` and `run_xa_temporal_fusion.py`: flow-guided,
  topology-constrained temporal mask repair with provenance checks.
- `tools/xa_vessel_topology.py`, `xa_vessel_depth.py`, and `xa_vessel_mesh.py`:
  graph extraction, underconstrained depth placement, and watertight mesh
  construction.
- `tools/build_dynamic_vessel_3d.py`: builds the time-varying research 2.5-D
  vessel scene in the supplied CT coordinate frame.
- `ctdr_pipeline/heart_mesh_motion/`: heart-surface projection, static pose
  reviews, synthetic motion fitting, and mesh quality checks. This is research
  code and does not claim that single-view XA uniquely recovers 3-D motion.

The differentiable heart-mesh experiment is isolated under
`experiments/heart_mesh_dynamic/`: its PyTorch/PyTorch3D pins differ from the
main registration environment, so it has its own `uv.lock` and environment.

## Safety and interpretation

XA geometry may lack an absolute patient reference frame. The vessel mesh uses
an explicit depth prior and remains a visualization hypothesis, not measured
3-D anatomy. Review source identity, projection direction, field of view,
temporal continuity, and independent landmarks before interpreting overlays.
Neither the current fit residual nor a visually plausible rendering is a
clinical accuracy measure.
