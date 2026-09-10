# GB10 GPU/Triton 首轮实现与实测结果

2026-09-10，使用最终代码在 NVIDIA GB10（48 SM）完成了
`collect controls → fit → collect holdouts → evaluate`。
这是一组用于验证跨后端架构的 pilot，**不是论文级 GPU 泛化结论**。

## 最终结果

| 项目 | 结果 |
| --- | --- |
| 独立 control | 36 点 |
| 按输入规模整组留出的 CV MAPE | 13.04% |
| 最大规模 CV fold MAPE | 24.83% |
| 组合 holdout | 15 点，全部计入 |
| holdout MAPE | 13.23% |
| holdout OOD | 0/15，限当前覆盖检查 |
| 常数对照：只取 control 延时中位数预测全部 holdout | 21.21% MAPE |
| 测量批次 | 55 批：51 批接受，4 批因波动拒绝并完整重测 |

校准门槛在测量前设为 **pooled leave-group-out MAPE ≤ 20%**，不是每个
fold 都必须低于 20%。最终最大规模 fold 为 24.83%，说明外推仍然薄弱；
不能用总体通过掩盖这个问题。holdout 规模处于 control 范围内部。

计时指标是 **CUDA graph 稳态缓存条件下的单 kernel 平均时间**，每张图
1,024 个 kernel、11 次 replay 取中位数，排除编译、capture 与 warmup。
不能称为冷缓存延时，也不能直接与 NKI 的 NC-p50 指标合并计算误差。

原始报告和完整测量在本机：

- [最终报告](../downloads/gpu_gb10_pilot_20260910/evaluation/report.json)
- [冻结校准与全部 CV fold](../downloads/gpu_gb10_pilot_20260910/calibration/frozen.json)
- [硬件、软件、源码指纹和 split](../downloads/gpu_gb10_pilot_20260910/manifest.json)
- [完整产物压缩包](../downloads/gpu_gb10_pilot_20260910.tar.gz)

这些实验产物位于被 Git 忽略的 `downloads/` 中，不随代码提交。
冻结校准内容摘要为 `6c89c297dc8530f97ade`，源码摘要为
`73b1ae22aa19836c53e3`。
已在 Inf2 本机直接重放 GPU 评估，得到完全相同的 13.2306635% MAPE；
同时核对了本地全部 Python 源码摘要与最终采集记录一致。

## 实现与论文主线的关系

已实现公共预测 API、公共规则数据结构与匹配器、公共 provenance digest，
并提供可复用的 control-only 校准、分组 CV 和 OOD 检查。NKI 保留原来的
校准与调度，通过兼容后端进入统一 API；GPU 有独立的 CPU Observe、工作量
展开、有效资源成本拟合和组合策略。详见
[架构与复现说明](performance_backends.md)。

GPU 规则目前根据源结构估计 sector、warp 算术、归约树与 dot 工作量。
尚未提供 GPU source-to-ISA 归因证据，不能宣称已经证明与 NKI 同等完整的
compiler-lowering 预测能力。拟合中的 ALU 有效系数为零，也不能解释为
GPU 算术指令没有成本：这组小规模控制实验不足以将其成本从其余开销中辨识。

softmax 三个点仍低估约 18%–27%，RMSNorm 低估约 14%–17%。这里只记录
结果，没有根据这些 holdout 调整常数、增设 operator lookup 或删点。
后续若扩展模型，需要独立的归约/布局/占用控制实验与新的预声明验证集。

开发中先前的 12.55% 结果已作废：字节数守恒测试发现 Triton 无 mask
访存委托给 masked 访存时出现双重回调计数，修复后重新采集并拟合了全部
control，再评估全部 holdout。本页只引用修复后的结果。

## 共享机器与验证条件

机器存在 GDM 的 Xorg 和 GNOME 常驻图形进程，显示未激活且初始利用率为
0%。连续三次空闲检查后，使用显式 `--allow-idle-graphics` 模式，结果标记
为 `monitored_shared_desktop`。测量前、中、后检查计算进程与新图形进程，
记录时钟、温度、功率和主机负载；没有修改其他进程、时钟或功率设置。
这减少了干扰风险，但不等于独占 GPU，也不能排除所有短时桌面活动。

环境为 Triton 3.7.0、PyTorch 2.9.1+cu128、驱动 580.95.05。
该 PyTorch 构建对 GB10 的 compute capability 12.1 发出兼容范围警告；
本次实际 Triton kernel 执行及输出校验通过，但不能据此推断整个 PyTorch
软件栈均支持这张卡。NVML 依赖装在独立临时虚拟环境，没有修改共享环境。

本地 85 项公共模型/NKI 回归测试通过，远端 27 项测试通过（两组有重叠）。
覆盖目标禁止编译、CPU/GPU 输出、mask 尾部、访存字节守恒、规则歧义、
control/holdout 隔离、校准摘要、OOD、共享机器检查以及 NKI 输出兼容。
代码通过 Ruff 检查和 diff whitespace 检查。
