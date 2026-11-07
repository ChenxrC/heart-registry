# CT-DR 配准流水线：Linux/NVIDIA 服务器运行说明

本项目把医生 PPTX 标注、病例目录中的 CT/定位像/XA，以及 DiffDRR 2D-3D 优化串成可审计、可断点续跑的批处理流程。所有输出均为科研候选；自动运行成功不等于临床级配准成功。缺少独立 3D 真值时，系统会把 3D TRE 留空并注明原因。

## 1. 输入结构

病例根目录的第一层文件夹名称必须以数字病例号开头，后面允许中文备注：

```text
/data/backup/1h1/
├── 1234568/
├── 1234567/
│   ├── CT1715346/
│   └── XA35397/
├── 1234501PRTct外院，本院有ct/
└── 1234502没有ct/
```

程序递归读取病例目录内 DICOM，但只用第一层目录名开头的数字作为病例号。PPTX 中第一页的病例号必须是可编辑文字框；程序不依赖奇偶页号，而是按病例号文字把连续页面归组。

## 2. 安装

已按服务器的 RTX 3090/3060、NVIDIA 驱动 535.309.01 配置 PyTorch CUDA 12.1 wheel：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd /data/work/registry
uv python install 3.11
uv sync --extra cuda121
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert torch.version.cuda.startswith('12.1')"
uv run ctdr version
```

如需 TotalSegmentator 椎骨/心脏 ROI，再执行：

```bash
uv sync --extra cuda121 --extra segmentation
```

TotalSegmentator 失败时会回退到骨密度 ROI，不会把分割失败伪装成成功。

## 3. 准备 PPTX 配准包

可在 Windows 本机生成后整体上传，也可直接在服务器生成：

```bash
uv run ctdr prepare-pptx \
  --pptx /data/input/标注-OKOK-pure.pptx \
  --output /data/work/bundles/标注-OKOK-pure \
  --overrides /data/input/annotation_landmarks.json
```

输出包含 `bundle.json`、`transforms.json/csv`、`landmarks.csv`、病例图片和 `landmark_transforms.xlsx`。JSON/CSV 是机器读取的权威数据，XLSX 用于人工查看。当前自动色块对应是保守策略；不能可靠对应的病例会进入 `review_required_2d`，不会捏造变换矩阵。

## 4. 扫描 DICOM 和人工覆盖

```bash
uv run ctdr scan-dicom \
  --data-root /data/backup/1h1 \
  --output /data/work/manifests/dicom.generated.yaml
```

若自动选择的 CT、定位像、XA 序列或帧不正确，复制并编辑 `configs/dicom_overrides.example.yaml`，再把路径写入运行配置的 `dicom_overrides`。覆盖文件只能引用 manifest 中真实存在的 Series UID，越界帧会直接报错。

运行前验证：

```bash
uv run ctdr validate \
  --bundle /data/work/bundles/标注-OKOK-pure \
  --dicom-manifest /data/work/manifests/dicom.generated.yaml
```

## 5. 单病例和批量运行

正式配置见 `configs/server.example.yaml`。可通过 `CUDA_VISIBLE_DEVICES` 控制两张卡的可见顺序；配置中的 `cuda:0`、`cuda:1` 指可见设备编号。

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run ctdr run-case \
  --case-id 1234567 \
  --bundle /data/work/bundles/标注-OKOK-pure \
  --dicom-manifest /data/work/manifests/dicom.generated.yaml \
  --config configs/server.example.yaml \
  --device cuda:0 \
  --output /data/work/runs/1234567

CUDA_VISIBLE_DEVICES=0,1 uv run ctdr run-all \
  --bundle /data/work/bundles/标注-OKOK-pure \
  --dicom-manifest /data/work/manifests/dicom.generated.yaml \
  --config configs/server.example.yaml \
  --output /data/work/runs/batch-001
```

`run-all` 默认在所有病例达到终态后退出 0，即使部分病例需要复核或缺 CT。加 `--fail-on-review` 后，只要存在非自动接受病例就退出 3，适合 CI/验收脚本。全局输入损坏或命令参数错误使用退出 2。

## 6. 断点续跑

每个病例的五个阶段为 `pptx`、`dicom_linkage`、`registration3d`、`evaluation`、`report`。每阶段保存 `stage.json`、输入/输出哈希、时间、警告和异常摘要：

- 输入和配置未变化：直接复用该阶段；
- 只改 DICOM override：从 `dicom_linkage` 起重跑，PPTX 阶段保持缓存；
- 只改质量阈值：只重跑 `evaluation` 和 `report`；
- 某病例失败：不影响其他病例。

不要手工修改阶段的 `payload.json`。如需强制重跑，使用新的输出目录最安全。

## 7. 结果与状态

```text
runs/<run_id>/
├── environment.json
├── summary.json
├── summary.csv
├── summary.xlsx
└── cases/<case_id>/
    ├── status.json
    ├── report.json
    ├── registration3d/candidates.csv
    ├── evaluation/metrics.json
    ├── evaluation/diagnostics/*.png
    └── stages/*/stage.json
```

关键状态：

- `accepted_research_candidate`：所有当前科研质量门均通过；
- `review_required_2d`：PPTX 点对应或仿射不可靠；
- `review_required_dicom_ambiguity`：定位像/XA 帧关联不唯一；
- `review_required_3d`：图像相似度、优化增益或多初值稳定性不足；
- `rejected_3d`：镜像、上下翻转、参数触边或视野 IoU 不合格；
- `skipped_*`：缺病例目录、CT、XA 或 PPTX 图像；
- `failed_runtime`：CUDA OOM、非有限梯度或其他运行时错误。

系统会比较最终 DRR 的原方向、上下翻转和左右翻转相似度。若翻转后明显更好，只记录为方向错误并拒绝，不会把翻转结果当成正确配准。

## 8. 报告重建与测试

不重跑配准，仅重建批次汇总：

```bash
uv run ctdr report --run-dir /data/work/runs/batch-001
uv run pytest tests/pipeline -v -m "not medical"
```

本机 CPU 快速冒烟配置是 `configs/local-smoke.yaml`，只验证链路，不代表正式精度。服务器正式运行使用 256/512 多尺度和更多初值。

## 9. 隐私与临床边界

默认报告过滤 PatientName、PatientID、出生日期、地址、医生姓名、机构名和 Accession Number；必要 UID 只保存 SHA-256。原始 DICOM 和含患者信息的 PPTX 仍属于敏感医疗数据，传输、权限、备份与删除应遵守医院制度。

当前单幅 XA 不提供每个标注点的独立深度，因此不能仅凭“切面看起来匹配”得到可信 3D 坐标。只有引入双投影、可靠设备标定、独立 3D 标志点或术中跟踪真值后，才能报告真实 3D TRE 并进一步讨论临床导航精度。

