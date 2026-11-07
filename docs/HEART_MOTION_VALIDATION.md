# CT–XA 心脏动态验证首轮：运行说明

P1 后续：逐帧复核、剔除导管、补选血管点及人工种子重新跟踪，见 [特征点复核操作说明](HEART_MOTION_REVIEW.md)。它不会自动提升标定或三维质量门控。

## 能做什么

此入口只做只读数据审计、未复核二维候选点跟踪、受约束合成点运动实验和深度歧义演示。不会修改现有分割、相位、标定或三维网页，也不执行患者真实三维非刚性重建。

路线与后续阶段见 `superpowers/specs/2026-10-17-ct-xa-heart-motion-validation-design.md`，逐项实施计划见 `superpowers/plans/2026-10-17-ct-xa-heart-motion-validation.md`。

## 输入要求

- `--case-dir`：一个病例目录，内部递归查找原始 CT/XA DICOM。
- `--assets-root`：该病例原有动态血管资产根，例如 `artifacts/xa_dynamic_vessel_1234567`。
- 原有资产中使用 `all_series/series_01`、`series_02` 的 `series_manifest.json`、`cine_u8.raw`、`temporal/temporal_centerlines_u8.raw`，以及原有标定、相位质量报告；不使用 candidate 分支。
- 心脏 mask 若存在，使用 `thorax_masks/heart.nii.gz` 的表面点作为合成形状。没有或无法读取时明确回退为数字点云，不能称为用了患者 CT 心脏运动。
- `--output-dir` 必须是全新目录，且不能位于原始病例或既有资产之内，不能是它们的祖先目录。已有空目录也拒绝，以免误覆盖。
- 候选目录和链接输入不属于权威数据；排除策略必须在哈希与实际读取时生效，不能只看报告警告。被排除的可选 mask 只允许明确回退数字点云。
- 观测累计时间须来自身份已匹配的源 DICOM 审计。旧缓存中的 `frame_times_ms`、`source_record.frame_time_ms` 或中位数可能只是播放默认值，不能据此认定实际采集时间。没有可信采集时间时保留逐帧空间轨迹，毫秒坐标为 null。

## Windows 当前项目环境

在项目根目录运行，输出目录名称每次更换：

```powershell
.\.venv\Scripts\python.exe run_heart_motion_validation.py --case-dir .\1234567 --assets-root .\artifacts\xa_dynamic_vessel_1234567 --output-dir .\artifacts\heart_motion_validation_1234567_20261017\run_001
```

## Linux / uv

复用本项目已有 `pyproject.toml`、`uv.lock` 和安装说明，不需要新增依赖或新训练环境。数据、旧资产和代码需同时可访问。等价入口为：

```bash
uv run python run_heart_motion_validation.py \
  --case-dir /path/to/data/1234567 \
  --assets-root /path/to/artifacts/xa_dynamic_vessel_1234567 \
  --output-dir /path/to/new_runs/heart_motion_001
```

以上 Linux 命令为部署接口说明；本轮先在本地 Windows 数据验证，不代表已连接服务器或完成 Linux 实测。

## 输出如何读

| 文件 | 含义 |
|---|---|
| `preflight.json` | 源哈希、CT/XA 几何/时间、既有质量门控与缺项 |
| `observations/series_01.json`、`series_02.json` | 自动点轨迹、有效性、前后向误差、丢失原因；坐标为列/行 |
| `observations/*_review.png` | 关键帧叠加。点 ID 表示跟踪 ID，不是解剖学分叉编号 |
| `review_template.json` | 空白人工复核记录，默认未复核；当前入口尚不消费填写后的模板来运行三维拟合 |
| `synthetic_benchmark.json` | 已知深度条件下的受限运动比较、留出误差和单视图深度歧义 |
| `summary.md` | 人可读的首轮结论与下一步 |
| `run_report.json` | 本轮范围、来源完整性、配置；`patient_3d_optimization_enabled=false` |
| `code_manifest.json` | 本轮代码 SHA256 与实际依赖版本，不虚构 Git 提交 |

自动跟踪结果始终待复核；有效点不是解剖正确的证明。缺失点不插值成观测。真实患者三维 TRE、EF 和心肌应变精度没有真值时不评价。合成实验的深度是已知条件，不是从 XA 自动恢复的结果。

直接调用 `build_observation_report` 而不经过实验入口时，需通过显式 `acquisition_timing` 契约提供可信的源审计，才会输出采集毫秒时间；只有未定类型的旧缓存时保持时间未知。实验入口负责源身份匹配及审计传递，不要求空间几何已接受才运行合法二维跟踪。

契约字段示例（供开发者，不应由未核实缓存自动生成）：

```json
{"source_verified": true, "source_link_status": "verified", "frame_count": 4, "timing_status": "available", "timing_source": "FrameTimeVector", "frame_timestamps_ms": [0, 100, 220, 300]}
```

时间戳已累计，不再二次求和；无审计、身份未核实、帧数不符、非有限/非递增时间则保留 N 个 null，并通过 timing_status/timing_provenance 说明原因。当前数据实际时间源是 DICOM FrameTime。

## 首轮软件验证与尚未验证的部分

2026-10-17 修正后：新增模块 focused suite 为203 passed、20 skipped；全项目回归868 passed、22 skipped，3条既有SWIG警告。20个新增跳过项是本机无权限创建的真实符号链接/junction，另2项为原项目已有同类跳过；模拟reparse元数据边界检查已执行，不能冒充真实链接测试。Linux/服务器及有权限的链接布局仍需补测。

当前病例运行在独立 `run_002`，490个输入哈希不变；它与保留的run_001轨迹完全相同，证明修正没有靠改观测制造更好结果。现有几何、相位、真实三维参考与临床门控依然全部未通过。代码通过测试与患者动态配准有效是两个不同结论。

第一批严格排除链接/候选布局。服务器若使用别名链接，应直接传入无链接的物理路径，或将所需输入放入普通目录；不降低输入权威性约束来绕过错误。

## 验证命令

测试临时目录也应使用全新路径，避免沙箱 AppData 临时目录访问问题。pytest 的 `--basetemp` 可能清理已存在目标，务必每次换名，不指向医院数据或产物目录。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/motion_validation -q --basetemp .\tmp\heart_motion_tests_fresh_001
```

旧模块使用顶层 tools 导入时，全量回归需在当前进程设置导入路径：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'tools') + [IO.Path]::PathSeparator + (Get-Location).Path
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\tmp\heart_motion_regression_fresh_001
```

首次原有代码基线：在项目内全新 basetemp 下 665 passed、2 skipped；默认沙箱临时目录曾导致权限错误，不把这种环境失败称为算法回归。
