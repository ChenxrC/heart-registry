# 心脏动态 Mesh：批次 A 使用与状态

本入口独立于旧血管重建。当前实现 `env`、`audit`、`synthetic`、`static-preflight`；真实 XA 心缘形变拟合和动态三维页属于后续批次，尚未启用。血管点不是前置。

最新静态修复结果见 `artifacts/heart_mesh_dynamic_1234567/static_preflight_004/summary.md`：优化网格工程检查已通过，S1/S2静态位姿仍为未验收候选。下文audit001/002为旧显示规则下的历史诊断，不被新口径改写。

## 已有产物

- `artifacts/heart_mesh_dynamic_1234567/audit_002/`：补强同字节输入校验后的真实病例静态投影。黄色是 CT 模型投影，**不是** XA 心缘分割或配准已通过。audit001 保留为早期记录。
- `projection_audit/series_01.png`、`series_02.png`：各7帧叠加。
- `projection_audit.json`、`mesh_report.json`：几何与优化网格状态；`mesh_reference.npz` 明确标记 optimization_ready。
- `coordination/`：测试、构建、审查和进度证据。
- 总结见 `artifacts/heart_mesh_dynamic_1234567/summary.md`。修正后的 `synthetic_smoke_002` 实际运行15组（5步/32分辨率），M1留出合成顶点RMS均值1.8756mm，M2为3.9308mm；都是数字模型而非患者TRE，M2尚未证明三维优势。动画仅是合成反演演示。

本机根环境为 torch2.5.1+cpu，无 PyTorch3D；隔离环境见 `experiments/heart_mesh_dynamic/README.md`，不修改根环境。必须以实际 env/backward 报告为准，不把库安装成功当作反演完成。

本次隔离 Windows CPU 的 PyTorch3D0.7.8 已编译成功，并实际通过反传探针、梯度有限差分和硬/软像素中心对照；不是 CUDA 验证，也未在服务器运行。输出只允许在上述专用结果根目录下创建一个**全新的直接子目录**；禁止病例目录、旧运行子目录、candidate 和链接路径。

## CPU 可用：真实数据只读投影审计

在项目根目录执行，每次使用全新目录：

```powershell
.\.venv\Scripts\python.exe -B run_dynamic_heart_mesh.py env --device cpu --output-dir artifacts/heart_mesh_dynamic_1234567/env_NEW
.\.venv\Scripts\python.exe -B run_dynamic_heart_mesh.py audit --case-dir 1234567 --assets-root artifacts/xa_dynamic_vessel_1234567 --audit-path artifacts/heart_motion_validation_1234567_20261017/run_002/preflight.json --output-dir artifacts/heart_mesh_dynamic_1234567/audit_NEW
```

env 不可用返回退出码2并写明原因。audit 是静态几何诊断，成功执行不表示标定/心缘匹配通过。程序核对当前选用的10个审计/影像/资产摘要；JSON/cine 对实际消费字节校验，NIfTI 从已校验的私有同字节临时副本解码，完成后再检查源路径摘要，不自动换用 candidate 或沿链接读取。此策略不等于操作系统级并发锁。

实际发现：原注册心脏表面256848面；压缩到10000面的候选虽保持较近的物理表面，却未通过既有角度质控，因此 `optimization_mesh_ready=false`。不放宽阈值。保留完整源表面**仅作静态投影**，不能凭该 NPZ 继续患者形变拟合。

S1/S2 约31.4%/39.7%的顶点投影落在图像内；这不是体积/表面积覆盖率。实际视野与候选相机均可能影响截断，不能单凭该数字推断标定错了多少。当前未取得独立 XA 心缘，因此重投影误差为 null、真实三维 TRE 为 null，未执行患者动态优化。

## 可微后端通过后：合成反演

```powershell
experiments\heart_mesh_dynamic\.venv\Scripts\python.exe -B run_dynamic_heart_mesh.py env --device cpu --output-dir artifacts/heart_mesh_dynamic_1234567/env_isolated_NEW
experiments\heart_mesh_dynamic\.venv\Scripts\python.exe -B run_dynamic_heart_mesh.py synthetic --smoke --steps 20 --resolution 32 --device cpu --output-dir artifacts/heart_mesh_dynamic_1234567/synthetic_smoke_NEW
```

后端不可用只生成 `benchmark_status.json`，不产生成功动画。smoke 是缩小配置的闭环检查：数字椭球、6帧、单平移案例、5种初始化、3组模型，不是完整基准或患者结果。只有实际执行优化后才导出 target/fitted 对照 GIF；禁止拿预设变形当反向恢复结果。

不加 `--smoke` 为12帧/10种案例/5初始化/3组模型，默认128分辨率、80步，运行量明显增加。当前合成求解使用完整 alpha BCE 和时间/幅度正则，真实局部心缘目标、自交验收、完整两级分辨率拟合仍待后续实现。报告保留限制；不宣称通过全套科学验收。

可选 `--reference-mesh <经审计且optimization_ready=true的NPZ>` 只把 CT 静态形状平移到**合成相机**，保存变换；不是在真实 XA 下恢复患者运动。当前 audit001/002 的未通过 NPZ 会拒绝。默认明确使用数字 Mesh，不伪装使用 CT。结果保存有效 bounds/loss_weights 和完整配置 SHA；非默认约束与数据项权重也实际参与优化。

像素中心约定：原生投影以整数索引为像素中心；PyTorch3D 光栅采用边界坐标，在 i+0.5 采样。软渲染适配在 resize 后将主点加0.5，仅转换坐标约定，不改变物理相机或经验平移模型。修复前两个独立对照均测得0.5px偏差，修复后通过；软 alpha 抗锯齿边界带不要求等同二值硬掩膜。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/heart_mesh_motion -q --basetemp tmp/heart_mesh_UNIQUE
```

实际 renderer 梯度测试需要编译好的 PyTorch3D；跳过不算通过。完整回归另需当前进程 `PYTHONPATH=<根目录>;<根目录>/tools`。原始数据、旧标定、旧血管页及既有患者分组不修改；本原型不用于临床导航、EF 或应变测量。

## 独立优化网格与逐序列静态初始化（用户已确认的新口径）

```powershell
experiments\heart_mesh_dynamic\.venv\Scripts\python.exe -B run_dynamic_heart_mesh.py static-preflight --case-dir 1234567 --assets-root artifacts/xa_dynamic_vessel_1234567 --audit-path artifacts/heart_motion_validation_1234567_20261017/run_002/preflight.json --ct-volume artifacts/annotation_registration/first_pair_1234567/totalsegmentator_depth/ct_series4_native_geometry.nii.gz --output-dir artifacts/heart_mesh_dynamic_1234567/static_preflight_NEW --steps 30 --resolution 64 --max-faces 20000 --device cpu
```

旧 `audit`/显示4°规则完全保留。新 `optimization_mesh_v1` 用带体积约束的二次误差简化；保留质心0.5mm、体积0.97–1.03、表面距离p95≤1.5mm，并另查闭合/一致朝向/退化/重复面、三角形质量、连通分量和Euler一致性、数值自交、原生像素可见轮廓p95≤1px、真实soft反传。夹角继续报告，不作为跨面数的绝对门槛。

自交采用VTK边界框筛选和双精度三角形相交测试；共享顶点对内缩1e-6排除正常公共边/点，具有明确数值容差，不是精确谓词证明。原始分割解剖拓扑未独立验证；当前源表面是一连通闭合分量、Euler=0，程序不会自动填掉其拓扑柄。`optimization_ready` 只表示当前工程数值检查通过，不表示心脏解剖/相机/临床验收通过。

静态校准第一阶段只拟合**探测器面内的两维毫米平移**，输出真正作用于CT坐标的4×4刚体矩阵及其逆；不是任意2D图像拖拽，也不是完整6DOF解。深度、旋转、内参、心脏形状锁定。相机和Mesh使用同一变换，旧标定不覆盖。骨性代理投影使用显式LPS affine/ray-box/grid-sample积分，单位毫米、原生像素中心；不是定量衰减模型，也不声称替代原DiffDRR物理校准。

S1/S2分别仅使用支持帧的中位背景，按时间变化降低运动/造影影响；留出帧不进入背景或位姿选择。该目标仍然含非骨组织，**不是人工骨标签**。搜索范围u/v±80mm、20mm粗网格，之后有限步梯度细化；图像得分升高不自动接受几何，尤其要防止错配重复肋骨/椎骨层级。

实际 `static_preflight_004`：2万面网格通过上述工程检查（相对原Mesh的采样表面距离p95约0.0348mm、体积变化约0.0326%、数值内部相交0、S1/S2模型轮廓差p95各1px、CPU反传非零）。S1/S2相似度约0.088→0.334、0.096→0.340，但**位姿未验收**：S2存在相距约45mm、分数几乎一样的粗搜索候选；S1也有多个相邻椎骨层级候选。不得把输出矩阵直接投入导航或动态形变。完整心脏不必落入冠脉局部视野，不能缩小心脏来迁就画面。

独立复核后补强了有效观测检查：在实际拟合分辨率/权重内，无目标变化或无投影变化时返回insufficient/null矩阵，而非靠NCC分母稳定化生成候选。004是修复后的重新执行；003保留为初轮。有效观测不等于唯一解，重复骨纹理的解剖歧义仍须另查。

该批输出 `mesh_optimization.npz`、质控JSON、逐序列4×4矩阵/候选相机、before_after图及源摘要。后续真实形变前仍需同名骨性对应/局部心缘复核与相机敏感性检查。当前无患者动态Mesh拟合、无独立三维TRE；原始数据只读。
