# heart-registry

面向研究的 CT–DR/XA 配准与三维重建代码库。主环境采用 `uv` 管理；入口既
可从 Python API 调用，也可通过命令行运行。仓库只保存源码、配置模板和
合成测试，不包含医院病例、医生标注 PPTX、模型权重或实验输出。

## 环境

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。CPU：

```bash
uv sync --extra cpu --extra dev
uv run ctdr --help
uv run pytest -q
```

Linux/NVIDIA CUDA 12.1 wheel：

```bash
uv sync --extra cuda121 --extra dev
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

若要做 TotalSegmentator 分割，再同步 `--extra segmentation`。动态心脏
可微网格实验是隔离环境，见
[`experiments/heart_mesh_dynamic/README.md`](experiments/heart_mesh_dynamic/README.md)。

若需要用椎体分割约束定位像关键点深度，安装 segmentation extra 后可运行
`uv run ctdr-landmark-depth --case-id <病例号> --ct-dir <CT目录> --ct-series <序列号> --summary <已审核配准摘要> --output-dir <新输出目录>`。
该深度是分割先验下的射线估计，不是单幅投影直接观测到的 3D 真值。

## 主要调用方式

### 命令行：PPTX 标注到 CT/DR 配准

```bash
uv run ctdr prepare-pptx --pptx /data/annotations/reviewed.pptx \
  --output /work/bundles/reviewed
uv run ctdr scan-dicom --data-root /data/cases \
  --output /work/manifests/dicom.yaml
uv run ctdr validate --bundle /work/bundles/reviewed \
  --dicom-manifest /work/manifests/dicom.yaml
uv run ctdr run-case --case-id 1234567 \
  --bundle /work/bundles/reviewed \
  --dicom-manifest /work/manifests/dicom.yaml \
  --config configs/server.example.yaml --device cuda:0 \
  --output /work/runs/case-1234567
```

`run-all` 对 bundle 中的所有病例执行断点续跑。输出目录必须与输入数据
分开；示例病例号仅为占位符。

### Python API

```python
from pathlib import Path
from heart_registry import prepare_annotation_bundle, scan_dicom_tree, register_case

bundle = Path("/work/bundles/reviewed")
manifest = Path("/work/manifests/dicom.yaml")
prepare_annotation_bundle(Path("/data/annotations/reviewed.pptx"), bundle)
scan_dicom_tree(Path("/data/cases"), manifest)
report = register_case(
    "1234567", bundle, manifest, Path("/work/runs/case-1234567"),
    Path("configs/server.example.yaml"), device="cuda:0",
)
print(report.decision.status)
```

其他公开函数见 `heart_registry.__all__`。XA 分割和动态网格也可以直接
从同一 API 调用；分割需要使用者另行取得并传入 DeepSA 源码和 checkpoint。

### 浏览器三维查看器

可复用的 CT/DR/XA 网页查看器源码位于 `web/ct_dr_3d_viewer/`，构建 wheel
时也会安装到 `share/heart-registry/web/ct_dr_3d_viewer/`。查看器读取流程生成的
病例几何与影像资产；这些患者级输出不会放入源码仓库。

### 命令行：XA 血管动态重建

所有数据路径都必须明确给出。下面展示步骤顺序；校准 JSON 和掩膜目录
应来自本病例经审核的输入，代码不会自动推断病例间对应关系：

```bash
uv run xa-audit-cine --xa-dir /data/case/XA --output /work/cine_manifest.json
uv run xa-extract-cine --dicom /data/case/XA/series.dcm --output-dir /work/series
uv run xa-segment-vessels --dicom /data/case/XA/series.dcm \
  --model-code-dir /opt/DeepSA --checkpoint /models/coronary.ckpt \
  --output-dir /work/series/segmentation --device cuda
uv run xa-temporal-fusion --dicom /data/case/XA/series.dcm \
  --segmentation-dir /work/series/segmentation \
  --output-dir /work/series/temporal
uv run xa-calibrate-series --dicom /data/case/XA/series.dcm \
  --reference /work/reviewed/calibration_report.json \
  --scene /work/reviewed/scene_geometry.json \
  --output /work/series/calibration.json
uv run xa-build-dynamic-mesh --series-dir /work/series \
  --masks-dir /work/ct/segmentation --temporal-dir /work/series/temporal \
  --calibration /work/series/calibration.json \
  --output-dir /work/series/dynamic_3d
```

The XA depth estimate is prior-driven because one projection does not determine
unique 3-D depth. Treat resulting meshes as research visualization, not as
patient-specific navigation truth.

## Architecture and limits

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and
[`docs/README_SERVER.md`](docs/README_SERVER.md) for batch server setup, status
codes, and output structure. This is not a validated medical device. Independent
3-D ground truth is required to report true 3-D TRE; optimizer residuals and
overlay similarity are not substitutes.
