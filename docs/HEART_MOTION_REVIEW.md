# CT–XA 动态方案：特征点复核与重新跟踪

这是既有 `HEART_MOTION_VALIDATION.md` 的 P1 后续工具，不是临床配准软件。目的是让医生剔除导管/背景候选，选择可追踪血管特征，并明确每一帧是否可见。不要求重做原有 PPT 骨性标记。

本地已生成：`artifacts/heart_motion_review_1234567_20261017/pack_002/index.html`。包含两段 XA 的 63 帧和 120 个待复核候选，真实人工标记仍为空。`pack_001` 为加固前软件包，保留用于审计，不作为当前操作入口；所有 `browser_qa_*` 下 JSON 都是自动化测试标记，不能作为医生数据。

## 1. 生成离线复核包

在项目根目录执行（Windows 将 `uv run python` 替换为 `.venv\Scripts\python.exe`）：

```bash
uv run python review_heart_motion.py build \
  --validation-run artifacts/heart_motion_validation_1234567_20261017/run_002 \
  --assets-root artifacts/xa_dynamic_vessel_1234567 \
  --output-dir artifacts/heart_motion_review_1234567_20261017/pack_NEW
```

输出目录必须全新，不能与输入重叠，不能使用链接或 `*_candidate`。程序核对旧审计中 cine、manifest、标定的 SHA256；变化时写前拒绝。真实病例生成 63 张原生灰度 PNG、自动轨迹、空白人工复核记录和几何检查。无需新依赖、网络、GPU 或上传服务。

双击输出 `index.html` 即可离线使用；`frames/`、`data.js`、`review.js` 必须一起保留。也可用仅绑定 `127.0.0.1` 的本地静态服务器。不要公开部署到互联网，目录含医院影像与来源标识。

## 2. 医生怎么操作

1. 选择 `series_01` 或 `series_02`，填写复核人编号。初始参考帧分别是零起始 20、16。
2. 灰色候选只是图像点，可能在导管上。点击附近候选或从列表选择；导管点选择“排除—导管”，不需要编造血管名称。
3. 对有用血管点，“确认当前建议坐标”或“点击修正位置”；候选不合适则“新增人工点”。不同血管特征应使用不同点 ID。
4. 填写明确的血管/分支名称，再选“纳入二维跟踪种子”。同一 ID 不同帧须是同一特征；不要把造影剂前缘/导管当作分叉，也不要把不同视角的同编号当作同一三维点。
5. 换帧检查，清楚时确认/修正，看不到时点“本帧不可见”或“不确定”。未标记帧仍未复核；纳入不会自动批准整条轨迹。不可见记录的坐标为 null。
6. 随时下载草稿，刷新不会自动保存；用“导入续审”恢复。完成后下载复核 JSON。工具只记录复核人自述，不认证身份。

坐标约定：原图列为 x、行为 y，零起始，向右/向下为正。页面缩放不改变坐标，不隐式翻转。播放仅是预览速度，不推断心动相位；可信采集时间另行显示。

## 3. 从明确纳入点重新跟踪

```bash
uv run python review_heart_motion.py retrack \
  --review /path/to/heart_motion_review_1234567_submitted.json \
  --bundle-dir artifacts/heart_motion_review_1234567_20261017/pack_NEW \
  --assets-root artifacts/xa_dynamic_vessel_1234567 \
  --output-dir artifacts/heart_motion_review_1234567_20261017/retrack_NEW
```

程序验证病例、pack_id、源哈希、点来源、帧范围、坐标、可见性和重复记录。默认空白草稿输出 `no_included_manual_seeds`，不会回退到所有自动候选。

每个纳入点取已明确确认的一帧为种子：优先原参考帧，否则最早可见人工帧。沿用原双向 LK 及质量阈值，不放宽、不按最近点重新找 ID。人工不可见/不确定帧将该帧建议置空；其他人工修正单独保留，不被预测覆盖或自动当作重检测种子。

输出 `retracking_report.json`：

- `manual_observations`：人工明确记录，单独保留。
- `tracked_candidates`：从人工种子传播的建议，仍为 `unreviewed_tracking_suggestion`。
- `review_summary`：纳入、排除、待定和明确观测数量，不等于准确率。
- `input_sha256`：实际输入的哈希。该机制用于溯源，不是带签名防篡改或身份认证系统。

没有医生复核文件时，不用程序测试标记冒充医生结果。始终不据此计算真实三维 TRE、EF、应变，不启用患者非刚性优化。

## 4. 几何检查如何理解

页面显示旧标定基底正交性、行列式、源到探测器/等中心距离误差、标定转移来源。数学自洽不等于解剖对齐。

本病例两段旧相机的正交误差约 `2.22e-16`，行列式约 `1`，源距离残差近数值零。这只说明参数内部一致。来源 Series 7 的 `6.6488 px` 不能填入当前两段冠脉的误差栏目。

当前旧标定由 Series 7 转移。来源序列 RMS 不是当前冠脉序列误差；当前序列实测重投影误差保留 null，不提升几何门控。真正改善定位仍需独立的本序列 CT–XA 对应点和留出评价；二维血管点没有唯一三维深度，重新跟踪本身不完成绝对标定。

## 5. 验证命令

```bash
uv run python -m pytest tests/motion_validation/test_review.py -q --basetemp tmp/review_UNIQUE
node --test tests/motion_validation/review_ui.test.cjs
```

Node 仅用于前端开发验证，运行页面不需要 Node。pytest basetemp 须全新，避免误清理已有目录。页面自动化测试使用专门测试复核人，只在测试目录保存，不作为真实医生结果。
