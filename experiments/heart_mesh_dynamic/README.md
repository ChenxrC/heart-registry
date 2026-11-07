# Isolated heart-mesh experiment environment

This is a separate `uv` project because the differentiable mesh renderer pins
Python 3.11, PyTorch 2.4.1, torchvision 0.19.1, and PyTorch3D 0.7.8
(`75ebeeaea0908c5527e7b1e305fbc7681382db47`). Keep its environment and lock
file separate from the main CT/DR registration package.

CPU bootstrap (PyTorch3D compiles from its pinned upstream source):

```bash
uv sync --project experiments/heart_mesh_dynamic \
  --extra cpu --extra render --no-install-package pytorch3d
uv sync --project experiments/heart_mesh_dynamic \
  --extra cpu --extra render --no-build-isolation-package pytorch3d
uv run --project experiments/heart_mesh_dynamic --no-sync \
  python run_dynamic_heart_mesh.py env --device cpu
```

For NVIDIA, select `--extra cuda121` instead of `--extra cpu`; ensure the
installed compiler matches `torch.version.cuda`. The CUDA version printed by
`nvidia-smi` describes driver capability, not an installed `nvcc` toolkit.
On Windows, native extension builds require the Visual Studio x64 build tools.

No case images, segmentations, meshes, checkpoints, or benchmark outputs are
included. Synthetic commands may be run with an explicit new output directory.
The code remains research-only; an available renderer or successful synthetic
fit is not evidence of patient-specific 3-D accuracy.
