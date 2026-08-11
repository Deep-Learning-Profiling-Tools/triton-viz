下面的分析以你提供的代码快照为准，并以 `downloads/Status.md` 中截至 **2026 年 8 月 9 日**的最新进度作为当前口径。由于该文档包含大量历史迭代记录，我会区分：

- **当前仍在使用的正式路径**
- **已经被后续工作取代的历史结果**
- **代码里仍然保留的兼容/实验性路径**

---

# 一、总体结论

当前 NKI 部分已经不是简单的“事件数 × 常数”模型，而是一套初步成形的、面向 Inf2/NeuronCore-v2 的编译器感知性能建模框架：

```text
NKI source
  → Triton-Viz CPU trace
  → source-level memory/compute events
  → Region IR / AccessPattern
  → compiler lowering / elimination
  → per-engine calibrated work
  → dependency/resource scheduler
  → runtime/sequencer model
  → NC-p50 latency prediction
```

当前实现最准确的定位是：

> 一个面向单 NeuronCore、有限 NKI grammar、由 Inf2 实测数据标定的 compiler-aware cost model 原型。

它目前最强的部分是：

1. **DMA、VectorE、ScalarE 已有较完整的真实硬件标定。**
2. **trace 和硬件验证可以使用同一份 Tilebench NKI kernel source。**
3. **已经认识并部分建模 compiler lowering、load CSE、Static DMA、stride 等 source/ISA 差异。**
4. **Load→SBUF→Compute→Store 的 storage/range/version 依赖已经贯通。**
5. **当前机制级模型在预注册的 35 个 FP32、`rows=128` formal holdout 上，可信 headline 是 NC-p50 MAPE 约 14.171%。**

但它还不能泛化地称为“通用 NKI cost model”，主要原因是：

- partition count 泛化仍不稳定，特别是 `p=1`；
- Static DMA、strided DMA 仍有 whole-kernel override；
- lowering 只主要预测 VectorE/ScalarE 聚合工作，缺少完整 ISA 顺序和 Flow；
- TensorE/GpSimdE/PSUM pipeline 仍不成熟；
- runtime 模型中部分参数不可辨识；
- OOD、compiler fingerprint 和 calibration compatibility 尚未真正进入生产门控；
- 当前三阶段复现 pipeline 的代码存在几处确定性错误和文档不一致。

---

# 二、整体代码架构

## 2.1 总体数据流

项目中实际上存在三条互相关联的管线。

### 1. Trace 管线

```text
NKI kernel
  → triton_viz.trace(frontend="nki" / "nki_beta2")
  → NumPy-backed interpreter
  → ClientManager callbacks
  → Load / Store / Transfer / Dot / NkiCompute records
  → trace.jsonl
```

### 2. 硬件测量管线

```text
NKI kernel
  → nki.benchmark
  → NEFF + NTFF
  → Neuron Explorer summary/parquet
  → engine active time / instruction / DMA packet / HBM bytes
  → all_results.csv / mapping CSV
```

### 3. 建模管线

```text
trace.jsonl
  → memory geometry / Region IR
  → compiler elimination
  → lowering expansion
  → calibrated event costs
  → dependency/resource scheduler
  → predicted engine busy time / NC latency
```

---

# 三、NKI Trace 与 Simulation 层

## 3.1 两套 NKI frontend

项目中存在两套 NKI 路径，这一点非常重要。

| 路径 | 主要文件 | 适用场景 |
|---|---|---|
| Legacy `nl.*` NKI | `core/simulation/nki.py`、`core/frontend/nki.py` | Tilebench、真实 `@nki.jit` source、当前 operator validation 主路径 |
| NKI Beta 2 / NISA | `core/simulation/nki_beta2.py`、`core/frontend/nki_beta2.py` | 显式 SBUF/PSUM/NISA 语义、底层 microbenchmark、beta2 examples |

---

## 3.2 `core/simulation/nki.py`

这是当前 Tilebench operator trace 的核心解释器。

它主要完成四件事。

### 1. NumPy 功能解释

`Builder` 模拟了：

- `nl.load`
- `nl.store`
- `nl.matmul`
- `nl.sum/max/min/mean`
- `nl.add/subtract/multiply/divide`
- `nl.exp/log/sigmoid/rsqrt`
- `nl.where`
- `nl.broadcast_to`
- 各种 elementwise/reduction

它的目标是：

> 保证 source-level 功能语义和可追踪性，不模拟硬件时序。

### 2. AST rewrite

`core/frontend/nki_transform.py` 将：

```python
nl.load(x[idx], mask=mask)
```

改写为：

```python
nl.load(x, idx, mask=mask)
```

类似处理：

- `nl.store`
- `nl.load_transpose2d`

这样解释器可以显式获取 base tensor 和 indexing keys。

### 3. Storage identity

`NDArray` 现在持有：

```python
_StorageState(
    identity,
    version,
)
```

并提供：

```python
storage_id()
tensor_version()
byte_range()
```

view、slice、reshape、broadcast 在共享 NumPy storage 时复用 `_StorageState`，从而支持：

- view alias；
- byte-range dependency；
- tensor version；
- in-place write 后的新版本。

### 4. Compute metadata tagging

`Builder._tag()` 给结果 NDArray 附加：

```python
_nki_api
_nki_engine
_nki_inputs
```

随后 `Tracer` 将其转成通用 `NkiCompute` record。

---

## 3.3 `core/frontend/nki.py`

这个文件把实际 `nl.*` API 映射到统一的 Triton-Viz op 类型。

例如：

```text
nl.load       → Load
nl.store      → Store
nl.matmul     → Dot
nl.sum        → ReduceSum
nl.exp        → NkiCompute
nl.add        → NkiCompute
nl.maximum    → NkiCompute
nl.rsqrt      → NkiCompute
```

它已经解决了早期一个非常关键的问题：

> softmax/rmsnorm/layernorm 中大量 `nl.*` compute 以前根本不会进入 trace。

目前真实 softmax 可以产生：

```text
Load
ReduceSum
NkiCompute(max/subtract/exp/divide/add/...)
Store
```

---

## 3.4 `core/simulation/nki_beta2.py`

Beta2 解释器更接近显式 NISA 语义，支持：

- HBM/SBUF/PSUM；
- `dma_copy`
- `dma_transpose`
- `tensor_copy`
- `tensor_tensor`
- `tensor_scalar`
- `tensor_reduce`
- `activation`
- `reciprocal`
- `exponential`
- `nc_matmul`
- `nc_transpose`

它还有大量 compiler parity 和合法性检查，例如：

- partition 不超过 128；
- matmul stationary/moving tile 上限；
- PSUM/SBUF engine 约束；
- dtype 组合；
- tile position/size；
- undefined-use；
- mutation 规则。

从测试数量看，Beta2 功能解释层已经相当扎实。

但要注意：

### Beta2 的性能事件覆盖仍明显不足

`core/frontend/nki_beta2.py` 目前只正式截获：

```text
program_id
ndarray
nc_matmul
dma_copy
dma_transpose
tensor_copy
tensor_tensor
```

没有正式转成 performance event 的操作仍包括：

```text
tensor_scalar
tensor_reduce
activation
reciprocal
exponential
nc_transpose
```

所以：

> Beta2 解释器“支持执行”不等于 Beta2 tracer“完整记录性能工作”。

这对以后从 beta2 examples 扩展到 RMSNorm、Softmax、Attention、GEMM 是一个重要缺口。

---

# 四、统一 Trace Record 层

## 4.1 `core/data.py`

核心性能 record 包括：

- `Grid`
- `Load`
- `Store`
- `Transfer`
- `Dot`
- `BinaryOp`
- `ReduceSum`
- `NkiCompute`

其中当前最重要的扩展是双侧 storage metadata。

### Load

现在同时记录：

```text
HBM:
  src_ptr
  src_storage
  src_range
  src_version
  src_dtype

SBUF:
  dst_ptr
  dst_storage
  dst_range
  dst_version
```

### Store

同时记录：

```text
SBUF:
  src_ptr
  src_storage
  src_range
  src_version
  src_dtype

HBM:
  dst_ptr
  dst_storage
  dst_version
```

### Compute

记录：

```text
input_ptrs
input_storages
input_ranges
input_versions

output_ptrs
output_storages
output_ranges
output_versions
```

这使 scheduler 可以建立：

```text
HBM Load
  → SBUF version 0
  → Compute
  → SBUF output/version
  → HBM Store
```

而不是只依赖不可靠的裸 pointer。

---

## 4.2 `clients/tracer/tracer.py`

Tracer 负责把 frontend callback 转成 dataclass records。

值得肯定的改进有：

1. `nl.load` 使用 after callback，能够读取实际返回的 SBUF NDArray。
2. `nl.store` adapter 传入 value，因此可以记录 SBUF source。
3. Dot 保留原始 stationary pointer，不再因 `.T` 创建伪 storage。
4. in-place assignment 会将已经生成的 compute record 重定向到真实 destination storage/version。
5. Load/Store bytes 按 active mask lanes 计算。

---

# 五、Trace Dump、Region IR 和 Source Mapping

## 5.1 `nki_trace_dump.py`

这个模块把 dataclass records 转成适合 cost model 的 JSONL event。

它补充的信息包括：

### DMA geometry

```text
partition_count
free_bytes_per_partition
partition_axis
dma_pattern
free_stride_items
partition_stride_bytes
active_access_count
access_span_bytes
access_density
```

### Compute metadata

```text
engine
free_dim
elements
flops
input/output storage
input/output byte range
input/output version
```

### Static DMA grouping

连续的小型 SBUF→SBUF scalar copy 被分组为：

```text
static_dma_group
static_dma_group_copies
static_dma_group_x
static_dma_group_y
```

### Source fusion region

同一个 `grid_idx` 中连续的 compute op 会形成一个 source region：

```text
fusion_signature
fusion_pattern
fusion_group
source_region_id
region_ir
```

Memory event 会切断 region。

需要强调：

> `fusion_group` 是 source-level lookup region，并不等价于“编译器一定把这些 op 融合了”。

这是正确的抽象边界。

---

## 5.2 `nki_region_ir.py`

Region IR 用于把完整 source signature 转成更可组合的结构特征。

当前包括：

```text
tokens
op_histogram
reduction_kind/count
broadcast_edge_count
partition_broadcast_input_count
one/two-input elementwise count
transcendental count
dtype
partition_count
free_dim / logical_free_dim
free_block_count
mask/tail
memory space
DAG edges
previous/next region family
```

随后通过声明式 `GrammarRule` registry 分类。

当前主要 grammar family 包括：

```text
reduction_rsqrt
two_reduction
two_reduction_rsqrt
reduction_transcendental
reduction_broadcast
elementwise_broadcast_multiply
elementwise_broadcast_affine
elementwise_multiply_n*
elementwise_mixed
elementwise_one_n*
elementwise_two_n*
```

这一步是当前工作中最有研究价值的部分之一，因为模型特征不依赖：

- operator 名；
- case 路径；
- allocation pointer；
- Tilebench benchmark 名。

---

## 5.3 `nki_instruction_source_mapping.py`

该模块把 Explorer instruction 映射回 source region。

当前主要证据链是：

```text
Instruction.penguin_id
  → compiler_artifacts/penguin.py op id
  → Penguin op class/opcode/tensor name
  → transfer-bounded source fusion region
```

附加启发式包括：

- 单 region 时间包络；
- 唯一 Activation/where owner；
- 唯一 reduction setup owner。

输出：

```text
instruction_mapping.csv
audit.json
```

Audit 中按 engine 给出：

```text
instruction_count
mapped_instruction_count
explorer_active_ns
instruction_union_ns
mapped_active_ns
payload_active_ns
mapped_payload_active_ns
region active-time union
```

采用 interval union 而不是 instruction duration 求和，避免并行指令重复计时，这是正确的。

---

# 六、Microbenchmark 框架结构

## 6.1 目录

```text
microbench/inf2_nki/
├── common/
├── configs/
├── harness/
├── profile_parser/
└── tests/
    ├── latency_pointer_chase/
    ├── bandwidth_dma/
    ├── engine_ops/
    ├── overlap/
    ├── program_mapping/
    ├── runtime_overhead/
    ├── static_dma/
    └── region_controls/
```

---

## 6.2 统一执行入口

`harness/run_microbench.py` 的 `BENCHMARKS` 注册：

```text
pointer_chase
dma_roundtrip_latency
dma_bandwidth
dma_strided_store
dma_transpose
dma_transpose_pipeline
vector_add
scalar_exp
tensor_matmul
tensor_dma_overlap
program_mapping
static_dma_scatter
runtime_overhead
```

每个 factory 返回：

```python
kernel, input_shapes, grid
```

随后执行：

```python
nki.benchmark(
    warmup=...,
    iters=...,
    save_neff_name="file.neff",
    save_trace_name="profile.ntff",
)
```

---

## 6.3 每个 case 的 artifact

典型目录包含：

```text
manifest.json
file.neff
profile.ntff
compiler_artifacts/
explorer_summary.json
explorer_parquet/
stdout.txt
stderr.txt
```

run 级别还包含：

```text
run_manifest.json
results.jsonl
```

其中 manifest 已加入：

- compiler/package/tool 版本；
- Inf2 hardware 信息；
- git revision；
- dirty worktree digest；
- Region IR schema version。

---

## 6.4 CSV 导出

`profile_parser/export_csv.py` 将所有 manifest 和 Explorer summary 扁平化为：

```text
all_results.csv
```

并派生：

```text
derived.read_gbps_dma_active
derived.write_gbps_dma_active
derived.read_gbps_dynamic_dma_active
derived.tensor_tflops_active
derived.vector_gelem_s_active
derived.*_byte_count_match
```

优点是新 benchmark 参数通常不需要修改固定 schema。

---

# 七、当前 cost model 的正式结构

## 7.1 预处理阶段

当前正式 replay 大致执行：

```python
events = load_trace()

events = eliminate_redundant_hbm_loads(events)

events = _expand_lowering_groups(events, model)

result = simulate(events, model)
```

---

## 7.2 Compiler elimination

`eliminate_redundant_hbm_loads()` 模拟精确 load CSE。

匹配条件是：

```text
same grid program
same source storage
same exact byte range
same offsets shape
same bytes
```

遇到重叠 HBM store 时 invalidate。

它解决了真实问题，例如：

```text
layernorm source 两次 load 同一 input
compiler 实际只发一次 DMA
```

FP32 F512 中曾观察到：

```text
source read bytes:   528384
hardware read bytes: 266240
```

---

## 7.3 DMA cost

### 旧 surface 路径

```text
(partition_count, free_bytes_per_partition)
  → measured GB/s
```

然后：

```text
time = bytes / bandwidth
```

支持：

- read copy surface；
- write surface；
- transpose surface。

### 当前 formal pipeline 更偏向 affine 路径

```text
DMA time
  = kernel DMA startup（只收一次）
  + read_bytes × read_ns_per_byte
  + write_bytes × write_ns_per_byte
```

即：

```python
DmaAffineCalibration
```

它使用 Explorer 的：

```text
software_dynamic_dma_active_time
```

从而尽量排除 Static DMA。

---

## 7.4 Level-B compute cost

`ComputeCalibration` 建模单条硬件指令成本：

```text
t_instruction
  = startup_ns
  + logical_free_dim × ns_per_free_elem
```

key 是：

```text
engine
dtype
input_stream_count
```

例如区分：

```text
VectorE one-input FP32
VectorE two-input FP32
VectorE one-input BF16
VectorE two-input BF16
ScalarE EXP FP32/BF16
```

代表性的 2026-08-07 V0 拟合大致为：

```text
Scalar FP32:
  28.946 + 0.300754 × F ns

Vector FP32 one-input:
  74.234 + 0.379977 × F ns

Vector FP32 two-input:
  65.948 + 0.789836 × F ns
```

不同运行中 intercept 会略有变化，因此这些参数必须绑定 compiler fingerprint，不能脱离 artifact 直接硬编码。

---

## 7.5 Level-A lowering

Level-A 回答：

> 一个 source region 最终展开成多少 VectorE/ScalarE work？

当前正式路径主要使用：

```python
StructuredControlCalibration
```

输入：

```text
Region IR / structural calibration key
dtype
free dimension
```

输出：

```text
effective instruction count
actual ISA instruction count
fixed mapped/unmapped work
```

region 成本近似：

```text
region_time(engine)
  = effective_count(engine)
  × Level-B instruction_time(engine, dtype, streams, F)
  + fixed_work(engine)
```

---

## 7.6 Dependency/resource scheduler

`simulate()` 当前支持：

### Engine resource

- VectorE 单 slot；
- ScalarE 单 slot；
- TensorE 单 slot；
- DMA 使用 16 个 token；
- 一个 transfer 占用：
  ```text
  min(partition_count, 16)
  ```
  个 token。

### Hazard

基于：

```text
storage id
byte range
tensor version
```

处理：

- RAW；
- WAR；
- WAW；
- view alias；
- disjoint range overlap；
- versioned tensor。

### Cross-engine handoff

依赖跨 engine 时可加：

```text
cross_engine_sync_ns
```

### Runtime path

存在 runtime calibration 时：

```text
final_latency
  = max(
      dependency/resource scheduler makespan,
      runtime setup path
    )
```

runtime setup path 包括：

```text
sequencer base
engine activation
cross-engine sync count
log2(partition_count)
log2(free access count)
```

---

# 八、核心实验总结

## 8.1 Pointer chase

结果约：

```text
2267 ns / dependent HBM load
R² ≈ 0.9997
```

意义：

- 这是 NKI-visible dependent HBM latency；
- 包含 dynamic indexing、DGE、DMA、同步、consumer wakeup；
- 不是裸 DRAM latency。

当前未直接进入正式模型。

---

## 8.2 Serialized DMA roundtrip

结果约：

```text
2618 ns / 8 KiB store + 8 KiB load
R² ≈ 0.9999
```

同样主要用于理解 trigger、completion、handoff，不是当前 formal runtime 的直接参数。

---

## 8.3 DMA free-dimension sweep

历史代表点：

| Free bytes/partition | Aggregate bandwidth |
|---:|---:|
| 128 B | 约 40.8 GB/s |
| 2 KiB | 约 207.5 GB/s |
| 4 KiB | 约 220 GB/s |
| 32 KiB | 约 270.8 GB/s |

说明：

- 小 free dimension 被 descriptor、trigger、packet overhead 支配；
- 大 free dimension 才接近：
  ```text
  16 × 17 B/ns = 272 GB/s
  ```

后续 single-transfer 大 free sweep 中，p=128 的 8/16/24/32 KiB 约为：

```text
205.7 / 242.3 / 246.3 / 255.6 GB/s
```

差异表明：

> repeat、keepalive 和 setup 口径会显著影响测得带宽，不能混用不同 sweep。

---

## 8.4 Partition × free-dimension surface

主要结论：

```text
active DMA engines ≈ min(partition_count, 16)
```

而不是早期假设的：

```text
ceil(partition_count / 8)
```

这是当前普通 DMA 模型最可靠的硬件结论之一。

---

## 8.5 Write surface

p=128 代表结果：

| Free bytes/partition | Write bandwidth |
|---:|---:|
| 128 B | 约 106.1 GB/s |
| 4 KiB | 约 237.5 GB/s |
| 16 KiB | 约 266.2 GB/s |
| 32 KiB | 约 267.0 GB/s |

说明 write 必须和 read 分开标定。

---

## 8.6 DMA transpose

p=128、free=4 KiB：

```text
copy      ≈ 198.2 GB/s
transpose ≈ 114.6 GB/s
```

因此 transpose 不能复用普通 copy surface。

`transpose_only`、`store_only`、`transpose_then_store` 对照没有给出稳定的、可单独建模的巨大 handoff penalty。更合理的方向是 packet/Flow 分析，而不是增加固定常数。

---

## 8.7 Static DMA scatter

目标 workload：

```text
多个 [p,1] tensor_copy
构造 SBUF free-dimension transpose
```

显式 Static DMA surface 在 calibration grid 上将 MAPE：

```text
约 64% → 约 6%
```

但 off-grid holdout 仍约 41%，说明：

- packet threshold；
- 分段行为；
- compiler lowering；

不能由简单平滑 IDW 完整表达。

---

## 8.8 Strided store

独立 stride-2 store control，p=128：

```text
F=128  → NC-p50 约 187 us
F=512  → 约 701 us
F=2048 → 约 2839 us
F=4096 → 约 5740 us
```

这与 Tilebench interleave 几乎逐点吻合。

关键结论：

> stride-2 HBM store 不是普通 contiguous DMA，而会生成 Static DMA packet train。

---

## 8.9 TensorE

已有实测：

```text
FP32 ≈ 20.2 TFLOP/s
BF16 ≈ 80.3 TFLOP/s
```

但当前 `Dot` cost 仍基本是：

```python
200 ns + FLOPs / 90 TFLOP/s
```

这显然仍是 placeholder，特别是 FP32 会严重乐观。

---

## 8.10 ScalarE EXP

实测：

```text
dependent   ≈ 161.7 GElem/s
independent ≈ 169.0 GElem/s
```

当前 Level-B ScalarE 已经接入这些量级。

---

## 8.11 Compiler lowering

softmax 的核心发现：

```text
source compute events: 约 6 个
hardware:
  VectorE 约 23 条
  ScalarE 约 24 条
  另有 GpSimdE/TensorE/sync/activation 等指令
```

因此：

```text
1 source op ≠ 1 hardware instruction
```

这是整个架构从简单 event model 转向 lowering-aware model 的决定性证据。

---

## 8.12 Source mapping 和 norm 泛化

使用 source→ISA mapping 与 structured controls 后，曾取得：

```text
FP32 norm:
  VectorE busy MAPE ≈ 4.42%
  ScalarE busy MAPE ≈ 3.70%

BF16 norm:
  VectorE busy MAPE ≈ 2.42%
  ScalarE busy MAPE ≈ 0.09%
```

这说明 lowering grammar 对 covered domain 中的 compute work 已经相当有效。

---

# 九、当前结果应该如何解读

历史日志中出现过：

```text
33点整体 MAPE 0.803%
35点 MAPE 2.845%
33点 MAPE 3.356%
```

但这些结果使用了不同程度的：

```text
structural completion residual
(structural_key, shape) → completion
```

后续已经明确认为这种方式容易退化为 shape-matched lookup，并将它从正式生产路径删除。

因此当前可信 headline 应该是：

```text
35 个 FP32 formal holdout
rows = 128
8 个 operator
NC-p50 MAPE ≈ 14.171%
```

对应 operator：

```text
interleave
kl_divergence
layernorm
mul2
relu
rmsnorm
sigmoid
softmax
```

更大的探索矩阵中：

```text
120 个成功硬件点
其中 85 个为 partition/shape OOD
全部 120 点 MAPE ≈ 63.21%
```

这说明当前模型的低误差仍是窄域内结果，不能宣称跨 partition/shape 泛化。

Partition leave-one-slice-out：

| Held-out partition | NC-p50 MAPE |
|---:|---:|
| p=16 | 10.153% |
| p=64 | 10.745% |
| p=128 | 10.481% |
| p=1 | 20.785% |

主要结论：

> p=1 是当前 partition 泛化最明确的失败点。

---

# 十、当前代码中值得肯定的设计

## 10.1 Calibration 和 holdout 有明确目录隔离

```text
controls/
holdouts/
calibration/
evaluation/
```

这个设计很好，应继续强化为机器检查，而不只是目录约定。

## 10.2 同一 kernel source 同时用于 trace 和 hardware

`nki_operator_experiments.py` 动态加载 Tilebench：

```text
impl_nki.py
```

trace 使用：

```python
kernel.func
```

hardware 使用同一个 `@nki.jit` kernel。

这比 `nki_model_experiments.py` 中分别写 trace/hardware kernel 可靠得多。

## 10.3 不使用 operator name 作为主要 cost key

Region IR、grammar、AccessPattern 都朝正确方向发展。

## 10.4 编译失败、OOD、高误差点没有被静默删除

Status 对：

- SBUF compile failure；
- trace unsupported；
- F4096 mapping failure；
- p=1 泛化失败；

都进行了明确记录。这对论文可信度很重要。

---

# 十一、代码审查发现的主要问题

以下按优先级排序。

---

## P0-1：三阶段复现 pipeline 当前存在确定性命令错误

文件：

```text
triton_viz/tools/nki_cost_model_pipeline.py
```

`fit()` 中调用：

```text
nki_fit_structural_static_dma
```

时传入了：

```text
--dma-affine-read-csv
--dma-affine-write-csv
--dma-affine-write-bf16-csv
--compute-calibration-csv
--structured-control-csv
--structural-static-dma-csv
```

但实际 `nki_fit_structural_static_dma.py` parser 只接受：

```text
roots...
--output
```

因此当前 README 宣称的三条命令 pipeline 很可能会在 fit 阶段直接报 unknown arguments。

而且 pipeline 还把：

```text
calibration/static_dma.csv
```

同时当成输入和输出传入，形成逻辑上的循环依赖。

### 建议

当前脚本逻辑下，调用应该简化为：

```bash
python -m triton_viz.tools.nki_fit_structural_static_dma \
  <controls-root> \
  --output <calibration/static_dma.csv>
```

### 必须增加的测试

现在的 pipeline 测试只检查 dry-run 输出中是否包含某些字符串，无法检测 CLI 参数是否真实有效。

建议每个工具暴露：

```python
def build_parser() -> argparse.ArgumentParser
```

pipeline 测试直接对生成的参数执行：

```python
tool.build_parser().parse_args(args)
```

确保所有子命令在不跑硬件时至少能够通过 parser。

---

## P0-2：pipeline 采集矩阵与 README 的 35 点 headline 不一致

README 声称 formal FP32 holdout 是：

```text
8 operators × F={128,512,1024,2048}
+ interleave/layernorm/rmsnorm 的 F=4096
= 35 points
```

但当前 pipeline 实际收集：

```text
elementwise_fp32:
  5 operators × 3 F = 15

norm_fp32:
  2 × 5 = 10

softmax:
  1 × 4 = 4
```

FP32 总数只有：

```text
15 + 10 + 4 = 29
```

加 BF16 4 点后总数才是 33。

因此：

> 当前三阶段 pipeline 无法复现 README 声称的 35 点 FP32 headline。

### 建议

将 formal split 固化为一个 JSON，而不是散落在 Python 列表中：

```json
{
  "formal_fp32_v1": {
    "rows": [128],
    "operators": {
      "interleave": [128, 512, 1024, 2048, 4096],
      "kl_divergence": [128, 512, 1024, 2048],
      "layernorm": [128, 512, 1024, 2048, 4096],
      "mul2": [128, 512, 1024, 2048],
      "relu": [128, 512, 1024, 2048],
      "rmsnorm": [128, 512, 1024, 2048, 4096],
      "sigmoid": [128, 512, 1024, 2048],
      "softmax": [128, 512, 1024, 2048]
    }
  }
}
```

evaluate 阶段应断言：

```text
expected case count == actual case count == 35
```

否则 pipeline 必须失败。

---

## P0-3：masked Load/Store 的 byte range 计算可能严重错误

`nki_trace_dump.py::_byte_span()` 当前只接收：

```python
offsets, nbytes
```

不接收 mask。

而 legacy `masked_load()` 在 false lane 上会填充：

```python
np.iinfo(dtype).max
```

因此 masked tail 的 offsets 中可能包含极大 sentinel。

随后 `_byte_span()` 直接：

```python
lo = flat.min()
hi = flat.max()
```

会生成近似：

```text
[正常地址, 2^63)
```

的巨大 range。

这会污染：

- RAW/WAR/WAW；
- load CSE；
- disjoint tile overlap；
- view alias；
- F4096 masked block dependency。

而且它用：

```python
nbytes // offsets_count
```

估计 element width；当 active lanes 少于 offsets count 时，这个宽度也会错误。

### 建议

最低限度改成：

```python
_byte_span(offsets, masks, item_bytes)
```

只使用：

```python
active_offsets = offsets[masks]
```

更理想的 schema 是：

```json
"byte_ranges": [
  [0, 512],
  [1024, 1536]
]
```

即 coalesce active offsets 成多个精确 segment，而不是一个 bounding box。

### 必须增加的测试

```text
masked false lane 含 int64 max，不应扩大 range
tail block 的 src_range 只覆盖 active lanes
两个 stride access bounding box overlap、实际 segment 不重叠时可并行
```

---

## P0-4：scheduler 对 partial-overlap writer history 的处理不正确

当前写入后会执行：

```python
writers[key] = [
    old
    for old in writers[key]
    if not overlap(new, old)
]
writers[key].append(new)
```

如果：

```text
old writer: [0, 100)
new writer: [0, 10)
```

旧 writer 会被整个删除。

之后读取：

```text
[50, 60)
```

将找不到 old writer，导致缺失 RAW 依赖。

### 建议

实现真正的 interval subtraction：

```text
old [0,100)
new [0,10)

保留 old remainder:
  [10,100)
```

或者使用：

```text
storage → version → interval map
```

来保存每个 version 的 writer ranges。

这是 scheduler correctness 问题，应优先于继续调 runtime 参数。

---

## P0-5：`AccessPattern` 丢失 signed stride

`nki_features.py` 中：

```python
free_stride_items=max(0, int(...))
```

会把：

```text
reverse stride = -1
unknown/irregular
```

都压成 0。

而 `layout_family` 又把：

```python
free_stride_items in (0, 1)
```

视为 contiguous 候选。

结果 reverse access 在 density≈1 时可能被错误分类为 contiguous。

### 建议

字段改为：

```python
free_stride_items: int | None
```

保留正负号。

分类至少应有：

```text
contiguous
strided_positive
reverse
broadcast_stride0
irregular
empty
```

不要用 `max(0, ...)` 淹没语义。

---

## P0-6：structured calibration key 丢失了 mask/context 等关键结构

`structural_family()` 会把：

```text
_masked
__after_xxx
__before_xxx
```

加入 family。

但 `structural_calibration_key()` 只包含：

```text
rule_id
op histogram
arity
```

没有包含：

- mask/tail；
- previous/next family；
- free_block_count；
- partition count；
- partition broadcast geometry。

于是 masked/unmasked、single-pass/two-pass context 可能落入同一个 key。

更严重的是，`StructuredControlCalibration.predict_points()` 对同 key、同 F 的多行直接取：

```python
exact[0]
```

这意味着结果可能依赖 CSV 行顺序。

### 建议

新 key 至少包含：

```text
rule_id
normalized op histogram
arity
mask/tail
previous rule
next rule
free_block_count
partition-broadcast count
partition count/bucket
```

并对相同 key/F 的多个控制点：

- 检查 compiler fingerprint 是否一致；
- 取 median；
- 输出 variance；
- 若 opcode fingerprint 不同，则拒绝合并。

不能再使用“第一行”。

---

## P0-7：formal evaluation 中存在静默 dtype fallback

`ComputeCalibration.instruction_ns()` 会在找不到 BF16 行时回退：

```text
BF16 → FP32
```

这对开发兼容有用，但对论文 formal evaluation 很危险。

可能出现：

```text
BF16 预测实际用了 FP32 calibration
但输出仍标为 BF16 通过
```

### 建议

增加：

```python
instruction_ns(..., strict_dtype=True)
```

formal evaluate 默认：

```text
strict_dtype=True
```

并在输出中记录：

```text
calibration_match = exact / dtype_fallback / streams_fallback / missing
```

任何 fallback 都应标 OOD，不进入正式 BF16 headline。

---

## P0-8：compiler fingerprint 已采集，但模型加载时没有门控

`nki_provenance.py` 已经可以收集：

- Neuron SDK；
- neuronx-cc；
- Explorer；
- runtime；
- hardware；
- repository revision/diff；
- Region IR schema。

但：

- `ComputeCalibration.from_csv`
- `StructuredControlCalibration.from_csv`
- DMA calibration loaders

不会检查这些 calibration 是否来自同一个兼容 fingerprint。

因此用户仍然可以无提示混合：

```text
V0 Level-A
+ V1 Level-B
+ V2 DMA
```

### 建议

建立模型 bundle：

```text
model_manifest.json
├── compiler_fingerprint
├── region_ir_schema
├── calibration file hashes
├── calibration case IDs
├── train split hash
├── Tilebench revision
└── compatibility policy
```

CostModel 构建时必须：

```text
exact match
verified compatible
requires canary
incompatible
```

四选一，不能静默加载。

---

## P1-1：当前 DMA affine calibration 本质上是 p=128 模型

`DmaAffineCalibration._fit_direction()` 默认固定：

```python
partition_count=128
```

但 `_dma_cost_ns()` 在任何 partition event 上都会直接使用该 slope。

这意味着：

```text
p=1、p=16、p=64
```

的 transfer 也会使用 p=128 的 bytes slope。

这正是 partition OOD 的主要来源之一。

### 建议

改成：

```text
(direction, dtype, partition_count)
  → startup + ns_per_byte
```

或者更完整地：

```text
(direction, dtype, p, free_bytes, packet_count)
  → timing
```

短期至少测：

```text
p = 1, 2, 4, 8, 16, 32, 64, 128
F = 128, 512, 2048, 8192
dtype = FP32, BF16
direction = read/write
```

并采用 piecewise/log-space interpolation。

尤其应单独处理：

```text
p=1
p=2..16
p>16
```

而不是从 p≥16 长距离外推 p=1。

---

## P1-2：runtime model 仍然存在不可辨识参数

当前 runtime control 拟合项有：

```text
sequencer_base
vector activation
scalar activation
tensor activation
cross-engine sync
partition setup
DMA packet/free-width setup
```

但现有 controls 中：

- empty kernel 无法编译；
- 所有合法 kernel 都有 HBM IO；
- TensorE control 缺失；
- cross-engine sync 与 engine activation 共线。

Status 已经观察到：

```text
cross-engine sync 最优系数接近 0
DMA activation 被 base 吸收
```

因此不能把这些参数都解释成独立物理量。

### 建议新增正交 controls

```text
minimal valid kernel
load-only
store-only
vector-only
scalar-only
tensor-only
DMA+Vector independent
DMA→Vector dependent
Vector→Scalar dependent
Scalar→Vector dependent
Tensor→Vector dependent
不同 program count
不同 packet count
```

runtime 模型最好转成显式 DAG node：

```text
launch
  → engine startup
  → work
  → handoff
  → completion
```

而不是所有 runtime 工作都放在一个与 scheduler 取 `max()` 的平行路径中。

---

## P1-3：source lowering expansion 只有 engine aggregate，没有真实顺序

`_expand_lowering_groups()` 会把一个 region 替换成：

```text
一个 VectorE aggregate event
一个 ScalarE aggregate event
```

这两个 event 通常读取同一批 external inputs，但没有：

- Vector→Scalar；
- Scalar→Vector；
- GpSimd→Vector；
- Tensor→Scalar；

等 compiler Flow 边。

因此它们容易被 scheduler 视为可并行。

这解释了为什么：

- engine busy 可以很准；
- scheduler makespan 仍然不准；
- runtime path 需要补偿大量时序。

### 建议

Level-A 输出不应只是：

```text
{engine: effective_count}
```

而应逐步升级为：

```python
LoweredRegion(
    micro_events=[
        ISAEvent(engine, opcode_family, timing),
        ...
    ],
    edges=[
        producer → consumer,
        semaphore wait/set,
        ...
    ],
)
```

至少先从 Explorer `Flow.parquet` 导入 region 内跨 engine 边。

---

## P1-4：Static DMA 目前只进入 busy time，没有进入 timeline

`simulate()` 中：

```python
structural_static_ns = predict_ns(...)
```

之后只做：

```python
engine_busy["static_dma"] = structural_static_ns
```

没有创建 timeline entry，也不改变：

```text
makespan
dependency chain
DMA resource occupancy
```

因此：

> combined DMA busy-time 可以准确，但 scheduler 不一定真正调度了 compiler-generated Static DMA。

另外当前写法会覆盖，而不是累加已有 explicit static DMA busy：

```python
engine_busy[ENGINE_STATIC_DMA] = structural_static_ns
```

### 建议

把 compiler-generated Static DMA 展开成真实 micro-events：

```text
StaticDmaPacketTrain
  reads SBUF range
  writes SBUF/HBM range
  occupies static DMA resource
  has producer/consumer edges
```

然后正常进入 scheduler。

---

## P1-5：Strided DMA 仍是 whole-kernel override

当前 `StridedDmaCalibration.predict()` 返回整个 kernel 的 strided DMA 时间。

`simulate()` 再将这个总时间按原始 cost 比例缩放到所有 DMA event，包括普通 load。

这虽然能复现 interleave，但并不是真正的通用机制模型。

风险包括：

- load 和 strided store 被错误同比例缩放；
- 多种 stride 混合时无法分解；
- 无法表达 packet overlap；
- 无法迁移到 reverse/gather/scatter；
- dependency timing 虽被“摊回事件”，但事件级持续时间没有硬件证据。

### 建议

统一生成：

```text
AccessPattern
  → contiguous segments
  → transaction/packet count
  → DMA micro-events
```

成本拆成：

```text
trigger
+ packet_count × packet_overhead
+ payload_bytes / bandwidth
+ contention
```

interleave 不再需要 whole-kernel special surface。

---

## P1-6：engine busy 和 duration/occupancy 概念混在一起

当前模型把 Explorer active time 拟合成一个 duration，然后直接用于：

- dependency completion；
- resource occupancy；
- engine busy；
- timeline end。

但这些不总是同一个量。

特别是 TensorE、DMA pipeline 中，应区分：

```text
issue interval
completion latency
resource occupancy
active-time contribution
```

### 建议

引入：

```python
@dataclass
class OpTiming:
    issue_interval_ns: float
    completion_latency_ns: float
    resource_occupancy_ns: dict[str, float]
    active_time_ns: dict[str, float]
```

scheduler 使用：

- issue interval 决定下一条独立 op；
- completion latency 决定 consumer；
- occupancy 决定资源冲突；
- active time 用于和 Explorer busy 对比。

---

## P2-1：TensorE 仍然是 placeholder

`Dot` event 当前缺少：

- input dtype；
- accumulator dtype；
- output dtype；
- M/K/N；
- tile position/size；
- perf mode；
- stationary reuse；
- accumulation；
- PSUM bank；
- LoadStationary/MultiplyMoving 分解。

现有：

```text
90 TFLOP/s
```

对 FP32 与 BF16 一视同仁，显然不成立。

### 建议第一阶段

在 `Dot` record 中加入：

```text
input_dtypes
accumulator_dtype
output_dtype
M/K/N
tile_position
tile_size
perf_mode
accumulate
stationary_storage/version
```

然后至少使用已有数据：

```text
FP32 ≈ 20.2 TFLOP/s
BF16 ≈ 80.3 TFLOP/s
```

### 最终目标

分解成：

```text
Tensor LoadStationary pipeline
Tensor MultiplyMoving pipeline
PSUM accumulation resource
eviction/cast path
```

这是进入 GEMM/Attention 正式验证前的必要条件。

---

## P2-2：Level-A 只主要覆盖 VectorE/ScalarE

softmax 实际 lowering 还包含：

```text
GpSimdE
TensorE
sync
activation setup
```

但 `nki_fit_structured_controls.py` 只遍历：

```python
("vector", 2)
("scalar", 1)
```

因此其他 engine 的工作：

- 被忽略；
- 被 fixed work 吸收；
- 被 runtime path 间接吸收。

### 建议

Level-A target engine 至少扩展到：

```text
vector
scalar
gpsimd
tensor
static_dma
sync/control
```

即使暂时不为每个 engine 建完整 latency，也应先准确保存真实 instruction count/fingerprint。

---

## P2-3：Beta2 compute tracer 覆盖不足

应复用 `NkiCompute` 或新增 `NkiIsaCompute`，覆盖：

```text
tensor_scalar
tensor_reduce
activation
reciprocal
exponential
nc_transpose
tensor_copy engine
```

事件需要保留：

```text
api_op
explicit engine
dtype
input/output storage
shape
axis
reduce op
broadcast geometry
tile metadata
```

否则 beta2 softmax/rmsnorm/attention trace 会系统性少算。

---

## P2-4：legacy simulator 仍有功能正确性问题

### 1. `silu`/`gelu` 没有 `_tag`

虽然 frontend 将它们映射为 `NkiCompute`，但 Builder 返回的 NDArray 没有：

```text
_nki_api
_nki_engine
_nki_inputs
```

Tracer 会直接跳过。

### 2. `_reduce()` 中 mask 处理是 no-op

当前代码：

```python
data = np.where(mask, data, data)
```

真假分支完全相同。

对于 masked sum/max/min，应使用对应 identity：

```text
sum  → 0
max  → -inf
min  → +inf
mean → masked count
```

### 3. in-place slice retarget 使用整个 parent range

`NDArray.__setitem__()` 将 compute output range 重定向为：

```python
self.byte_range()
```

而不是实际 target slice range。

对部分 slice write 会制造过大的 WAW/RAW 范围。

---

## P2-5：Region IR 本身还有几个结构性问题

### 1. one-input elementwise 会把 reduction 重复计入

当前：

```python
one_input = token in _ONE_INPUT or arity == 1
```

`reduce_sum` 通常 arity=1，因此同时进入：

```text
reduction_count
one_input_elementwise_count
```

对 compositional model 会造成双重计数。

### 2. partition axis 默认使用 shape[0]

对非 leading partition axis、beta2 `par_dim` 或更复杂 layout 不成立。

### 3. partition broadcast 通过相邻 load 推断

当前可能把不真正 feed 当前 region 的 p=1 load 误判成 partition-broadcast input。

应基于 storage/version DAG 连接，而不是仅看邻近 memory window。

### 4. `source_region_id` 中实际 kernel name 经常退化为 `"kernel"`

如果不同 kernel 具有相同：

```text
region ordinal + signature + shape + dtype
```

可能生成相同 ID。

应加入：

- source file hash；
- qualified kernel name；
- Tilebench revision；
- source line范围。

---

## P2-6：Source mapping 仍有启发式风险

当前 `assign_penguin_regions()` 是 greedy chunk matching。

当多个 region 都有相似 token 时可能错配。

另外：

- time-envelope match confidence=0.7；
- unique-owner match confidence=0.8/0.9；
- fitter 不一定过滤低 confidence；
- 100% mapped coverage 不代表 100% attribution 正确。

### 建议

1. 使用 order-preserving dynamic programming，而不是 greedy。
2. 匹配目标同时考虑：
   - opcode multiset；
   - source order；
   - tensor names；
   - BIR/Penguin IDs；
   - transfer boundary；
   - Flow producer/consumer。
3. calibration 默认只接收：
   ```text
   confidence >= threshold
   ```
4. 对不同 match method 分别做 sensitivity analysis。

---

## P3-1：OOD 体系仍不完整

当前 grammar OOD 主要检测：

- unknown op；
- empty grammar；
- unsupported schema。

但没有检测：

- partition 是否在 calibration hull；
- F 是否外推；
- dtype 是否 fallback；
- free block branch 是否未见；
- context 是否未见；
- mapping coverage 是否不足；
- compiler fingerprint 是否不兼容；
- stride/density 是否未校准。

因此可能出现：

```text
Region IR in-scope
但 artifact lowering 无法解释
```

F4096 就是实际反例。

### 建议输出独立状态

```text
trace_capability_status
grammar_domain_status
mapping_coverage_status
compute_calibration_status
dma_calibration_status
runtime_domain_status
compiler_compatibility_status
final_prediction_status
```

正式 MAPE 同时报告：

```text
accuracy
coverage
OOD rate
fallback rate
```

---

## P3-2：Calibration loader 普遍存在静默 clamp

包括：

- `DmaCalibrationSurface`
- `LoweringExpansionCalibration`
- `StructuredControlCalibration`
- `StridedDmaCalibration`
- `StructuralStaticDmaCalibration`

多数在超出范围时直接使用边界点或 nearest neighbor。

这对工程 fallback 可以接受，但不能将结果标成 in-domain。

### 建议

每次 lookup 返回：

```python
CalibrationLookup(
    value,
    mode="exact|interpolated|clamped|fallback|missing",
    distance=...,
    source_points=...,
)
```

formal evaluation 只允许：

```text
exact
interpolated in hull
```

---

## P3-3：当前 cross-version 只完成基础设施，没有完成实验

已经有：

- fingerprint；
- lowering diff；
- same-version canary。

但还没有真正的：

```text
V0 compiler
vs
V1 compiler
```

paired experiment。

因此目前不能声称：

- grammar 跨版本稳定；
- cost-only adaptation 足够；
- canary 能低成本迁移。

需要在同一 clean commit、同一 Inf2 SKU、同一 source/config 下采集至少两个 compiler stack。

---

## P3-4：统计稳定性不足

当前很多结果使用：

```text
一次 compile/profile
NC p50
Explorer summary active time
```

但缺少：

- 重复 compile；
- 重复 profile；
- confidence interval；
- compiler nondeterminism 分布；
- active-time variance；
- p50 的整数微秒量化影响。

### 建议

正式论文数据至少：

```text
3 次独立 compile/profile
每次 N 次迭代
报告 median + IQR/CI
```

对同 compiler 的 fingerprint 差异先建立 false-positive baseline。

---

# 十二、三阶段 pipeline 还存在的其他复现问题

## 12.1 BF16 runtime fit 可能使用错误 write CSV

`nki_fit_runtime_overhead.py` 会按 dtype 构造 `DmaAffineCalibration`。

但 pipeline 只传：

```text
dma_write_fp32.csv
```

即使 runtime controls 中包含 BF16，也可能找不到 BF16 write rows。

应改为 dtype→CSV mapping，或分别拟合 runtime FP32/BF16。

---

## 12.2 Tilebench revision 没有进入 fingerprint

当前 experiment manifest 记录了 Tilebench 路径，但没有记录：

```text
Tilebench git revision
dirty state
impl_nki.py hash
```

“同一 kernel source”必须可验证，而不能只依赖路径。

---

## 12.3 operator 输入没有固定随机种子

`_randn()` 使用全局：

```python
np.random.randn
```

没有显式 seed。

虽然 cost 主要依赖 shape/dtype，但以下行为可能受输入影响：

- data-dependent branches；
- mask；
- DGE；
- special values；
- numerical control flow。

建议添加：

```text
--seed
```

并将输入 hash 写入 manifest。

---

## 12.4 文档存在多套互相冲突的复现方式

目前至少有：

- 顶层 README 三阶段 pipeline；
- `microbench/README.md`；
- `tools/README.md`；
- 中文 10 组复现指南；
- Status 历史日志。

其中有些仍使用：

```text
legacy softmax Level-A CSV
```

而后续正式路线已经改为独立 softmax reduction control 或 fresh mapped artifacts。

建议只保留一个 canonical protocol，其余明确标注：

```text
Historical / superseded
```

---

# 十三、建议的具体工作计划

---

## P0：先保证当前结果真正可复现、模型基础正确

### 任务 1：修复三阶段 pipeline

修改：

```text
triton_viz/tools/nki_cost_model_pipeline.py
```

完成：

1. 修复 `nki_fit_structural_static_dma` 参数。
2. 修复 BF16 runtime write calibration。
3. 将 formal 35 点 split 固化为 JSON。
4. evaluate 断言 case count。
5. 输出 FP32 formal、BF16 auxiliary、OOD exploration 三套独立指标。
6. 增加 parser contract test。
7. 从空目录实际跑一次 smoke pipeline。

验收：

```text
collect --dry-run
fit --dry-run
evaluate --dry-run
```

均能生成可执行命令，并且小型 fixture 的 fit/evaluate 能真正执行。

---

### 任务 2：修复 memory range correctness

修改：

```text
nki_trace_dump.py
nki_cost_model.py
core/simulation/nki.py
core/data.py
```

完成：

- mask-aware active offsets；
- item-size-aware range；
- multiple byte segments；
- partial-overlap interval subtraction；
- slice assignment 使用实际 target range；
- reverse/stride0/irregular AccessPattern。

验收测试：

```text
masked sentinel 不扩大 range
partial overwrite 后非重叠旧 writer 仍存在
reverse stride 不会分类为 contiguous
disjoint strided segment 可 overlap
view alias + partial overlap 正确
```

---

### 任务 3：收紧 structured key 和 calibration aggregation

修改：

```text
nki_region_ir.py
nki_fit_structured_controls.py
nki_cost_model.py
```

完成：

- key 加入 mask/context/free blocks/partition；
- 同 key/F 多点用 median；
- 保存 variance；
- fingerprint 不一致时拒绝合并；
- calibration lookup 返回 exact/interpolated/clamped；
- formal evaluation 禁止 silent fallback。

---

### 任务 4：建立正式 model bundle

新增：

```text
calibration/model_manifest.json
```

内容：

```text
compiler fingerprint
region IR version
Tilebench revision
all calibration CSV hash
train split hash
control case IDs
supported domain
known OOD
```

CostModel 加载时执行 compatibility gate。

---

## P1：解决 partition 泛化和 runtime 可辨识性

### 任务 1：补 partition controls

至少采：

```text
p = 1, 2, 4, 8, 16, 32, 64, 128
F = 128, 512, 1024, 2048, 4096
dtype = FP32/BF16
```

但不需要全笛卡尔积，可以对主要 grammar 采用 space-filling 设计。

优先 primitive：

```text
unary map
binary map
reduction
reduce+rsqrt
broadcast affine
contiguous read/write
strided store
```

### 任务 2：预注册 slice holdout

例如：

```text
train p={1,16,128}, test p=64
train p={1,64,128}, test p=16
train p={16,64,128}, test p=1
```

目标：

```text
每个 partition slice MAPE <15%
特别是 p=1 从 20.8% 降至 <15%
```

### 任务 3：重构 runtime controls

加入：

```text
tensor-only
engine pair independent/dependent
program count
packet count
handoff count
loop epochs
```

对不可辨识参数：

- 合并；
- 删除；
- 或明确标注 “joint term”。

不能继续给每个物理名一个实际上不可辨识的系数。

---

## P2：统一 DMA micro-event 模型

### 目标结构

```text
AccessPattern
  → transaction segmentation
  → packet count
  → DMA engine set
  → packet micro-events
  → scheduler
```

### 特征

```text
direction
dtype
partition start/count
free stride
partition stride
segments
density
alignment
span
packet count
repeat
transpose
gather/scatter
```

### 要替换的路径

逐步淘汰：

```text
whole-kernel StridedDmaCalibration override
StructuralStaticDmaCalibration whole-rule-sequence lookup
```

### 验收

```text
contiguous / strided / reverse / masked / scatter
共享同一 cost interface

两个 p=128 transfer 不超过单核带宽
两个 p=8 transfer 只在 engine set 允许时 overlap
interleave 不再需要 whole-kernel特例
combined DMA busy MAPE <10%
packet span error <15%
```

---

## P3：完成 compiler Flow 与完整 lowering

### 任务 1：扩展 mapping

读取：

```text
Instruction.parquet
Flow.parquet
DmaPacket.parquet
semaphore events
```

输出：

```text
region→ISA events
ISA producer/consumer edges
engine/resource occupancy
confidence
```

### 任务 2：扩展 Level-A target engine

加入：

```text
GpSimdE
TensorE
Static DMA
sync/control
```

### 任务 3：Beta2 compute coverage

截获：

```text
tensor_scalar
tensor_reduce
activation
reciprocal
exponential
nc_transpose
```

### 任务 4：修复 source region identity

加入真实：

```text
kernel qualified name
source file digest
source line range
Tilebench revision
```

---

## P4：TensorE/GEMM/Attention

在完成上述正确性工作后再正式扩展。

### TensorE microbench 矩阵

```text
dtype = FP32, BF16
M = 32,64,128
K = 32,64,128
N = 64,128,256,512
repeat = 1,2,4,8,16
mode =
  dependent accumulate
  independent
  same stationary
  alternating stationary
```

应分析：

```text
LoadStationary
MultiplyMoving
PSUM accumulation
stationary reuse
issue interval
completion latency
```

先验证：

```text
GEMM
```

再验证：

```text
attention score matmul
softmax
probability×V matmul
```

不建议直接跳到完整 Attention end-to-end，否则难以定位误差来源。

---

## P5：跨 compiler 版本实验

流程：

```text
V0 clean commit + compiler A
V1 same clean commit + compiler B
```

先跑 same-version repeat baseline，再跑 cross-version。

报告：

```text
same lowering
threshold drift
structural drift
cost-only drift
mapping/schema drift
nondeterministic
```

比较四种迁移：

```text
zero-shot
cost-only refit
canary/control-adapted
full refit
```

目标：

```text
control-adapted 不使用 V1 operator holdout
count MAPE <15%
busy MAPE <20%
采集成本比 full calibration 降低至少 50%
```

---

# 十四、建议的代码重构

`nki_cost_model.py` 当前已经过于庞大，混合了：

- calibration loader；
- preprocessing；
- lowering；
- cost；
- dependency；
- resource scheduler；
- runtime；
- CLI。

建议在 correctness 稳定后拆为：

```text
triton_viz/tools/nki_model/
├── schema.py
├── domain.py
├── provenance.py
├── preprocessing/
│   ├── load_cse.py
│   └── lowering.py
├── calibration/
│   ├── dma.py
│   ├── compute.py
│   ├── runtime.py
│   └── bundle.py
├── costs/
│   ├── dma.py
│   ├── vector.py
│   ├── scalar.py
│   ├── tensor.py
│   └── static_dma.py
├── scheduler/
│   ├── dependency.py
│   ├── intervals.py
│   ├── resources.py
│   └── simulate.py
└── evaluation/
    ├── metrics.py
    └── report.py
```

但不要为了重构而中断实验；应先用现有测试锁定行为，再逐模块迁移。

---

# 十五、最优先做的 10 件事

按收益和依赖关系排序：

1. **修复 `nki_cost_model_pipeline.py` 的 static-DMA CLI 错误。**
2. **让 pipeline 真正采集 README 所述的 35 个 FP32 formal case。**
3. **修复 masked offsets 导致的错误 byte range。**
4. **修复 scheduler partial-overlap writer history 丢失问题。**
5. **修复 signed stride/irregular AccessPattern 分类。**
6. **重做 structured calibration key，并消除 `exact[0]` 的任意行选择。**
7. **为 calibration lookup 增加 strict dtype、OOD、插值距离和 provenance 门控。**
8. **将 DMA calibration 扩展为 partition-aware，优先解决 p=1。**
9. **把 Static DMA/strided DMA 从 whole-kernel override 转成可调度 micro-events。**
10. **为 Dot 增加 dtype/tile/perf metadata，并接入真实 FP32/BF16 TensorE 数据。**

---

# 十六、最终评价

## 已经比较成熟的部分

- Inf2 microbenchmark harness；
- artifact 和 CSV 导出；
- directional DMA 实验；
- VectorE/ScalarE Level-B；
- source→Penguin/ISA mapping；
- structured Region IR；
- load CSE；
- storage/range/version dependency；
- 单 NeuronCore、p=128、covered grammar 的 compute busy prediction。

## 仍属于原型的部分

- partition 泛化；
- runtime 物理分解；
- Static/strided DMA scheduling；
- Flow/semaphore；
- TensorE/GpSimdE；
- compiler-version migration；
- OOD/uncertainty；
- low-level NKI frontend coverage；
- 多 program、多 core。

## 当前最合理的研究主张

可以主张：

> 对有限 NKI grammar，source-level region 可以通过独立 controls 映射成可审计的 per-engine work；结合方向化 DMA、storage dependency 和机制级 runtime control，可在单 NeuronCore、预注册 in-domain workload 上达到约 14% 的 NC-p50 MAPE。

暂时不宜主张：

> 该模型已经能通用预测任意 NKI kernel、任意 partition、GEMM/Attention、多 core 或任意 compiler 版本。

下一阶段最重要的不是继续添加 operator-specific 分支或调一个 overhead，而是：

```text
修复 pipeline 与依赖正确性
+ partition-aware calibration
+ packet/ISA micro-event lowering
+ compiler Flow
+ strict OOD/provenance
+ TensorE pipeline
```

这些完成后，模型才真正具备从“窄域内拟合良好”升级为“可扩展、可审计、可发表的 NKI compiler cost model”的基础。

---

# 十七、2026-08-10 P0 修复进度

本节记录 `Status_new.md` 中 P0-1 至 P0-8 的当前实现状态。以下改动均在
`/home/ubuntu/triton-viz` 工作树完成。

## P0-1：三阶段 pipeline 命令契约

状态：**已修复并测试**。

- `nki_fit_structural_static_dma` 调用已移除所有无效参数以及输入/输出循环依赖。
- pipeline 在执行每条子命令前，会用子工具的真实 argparse parser 验证完整 argv。
- `nki_fit_structural_static_dma` 已暴露 `build_parser()`，测试覆盖旧参数被拒绝。
- runtime fit 新增 `--dma-affine-write-bf16-csv`，按 dtype 选择 write calibration。
- `runtime_overhead.json` 移除已知 compiler-invalid 的 `empty` case，避免 collect
  因预期编译失败返回非零并中断整个三阶段 pipeline。
- collect 子命令现在默认使用 microbench `--skip-existing` 和 controls/holdouts
  `--resume`，长时间 Inf2 采集可在中断后继续。
- microbench harness 现在把 `skipped_existing` 计为成功；此前 resume 会在完整跳过
  第一个已完成 suite 后错误返回非零，导致 pipeline 无法继续到下一个 suite。

## P0-2：formal 35 点 split

状态：**已修复并测试**。

- 新增 `microbench/inf2_nki/configs/formal_holdouts.json`。
- `formal_fp32_v1` 精确包含 README 声明的 35 点。
- `auxiliary_bf16_v1` 与 FP32 headline 分离。
- evaluate 对 formal FP32 成功行数执行 `expected == actual == 35` 强断言。
- report 分开记录 FP32 formal 与 BF16 auxiliary case 数量。

## P0-3：masked byte range

状态：**已修复并测试**。

- byte span 只使用 active mask lanes，不再包含 false lane 的 int64 sentinel。
- item width 按 active lane 数或显式 item size 计算。
- trace 新增 `src_ranges` / `dst_ranges` 精确 coalesced segments，同时保留兼容 bounding range。
- scheduler 优先使用 exact segments，因此 bounding box 重叠但真实地址不相交的访问可并行。
- exact segment 数量上限为 1024；超大 interleave/scatter 不再把数万到数十万
  one-element segment 写入 JSONL，也不会让 interval history 进入二次复杂度。该类
  访问回退到 conservative bounding range，而 stride/density/active-count 仍保留给
  DMA calibration。小型和 tail/alias correctness case 继续使用 exact segments。

## P0-4：partial-overlap writer history

状态：**已修复并测试**。

- writer/read history 更新由“删除所有 overlap entry”改为 interval subtraction。
- partial overwrite 后保留旧 writer 未覆盖的左右 remainder。
- 测试覆盖 old `[0,100)`、new `[0,10)` 后读取 `[50,60)` 仍依赖 old writer。

## P0-5：signed stride

状态：**已修复并测试**。

- `AccessPattern.free_stride_items` 和 partition stride 保留 signed/unknown 语义。
- layout family 现在区分：
  `contiguous`、`strided_positive`、`reverse`、`broadcast_stride0`、
  `irregular`、`empty`。
- reverse stride 不再被归类为 contiguous。

## P0-6：structured calibration key 与重复点

状态：**已修复并测试**。

- calibration key 新增 mask/tail、previous/next context、free block count、
  partition broadcast count、partition count/bucket。
- structured-control 导出新增 opcode fingerprint、replicate count 和 variance。
- 相同 key/F 重复点使用 median，不再依赖 CSV 第一行。
- compiler version 或 opcode fingerprint 不一致时拒绝合并。

## P0-7：formal dtype fallback

状态：**已修复并测试**。

- `ComputeCalibration` 新增带 match-kind 的 strict lookup。
- formal FP32 replay 启用 `--strict-calibration`；missing exact calibration 直接失败。
- replay CSV 新增 `calibration_match`，记录 exact/fallback/missing 类型。
- predicate-producing events（例如 `greater`）即使 `output_dtype=bool`，也使用其
  non-bool input/Region IR value dtype 查 calibration；这不是 dtype fallback，BF16 与
  FP32 仍严格分离。
- formal FP32 与 auxiliary BF16 replay 都启用 strict calibration；audit 改为检查
  实际 lowered events，不再把已被 structured lowering 替换的 raw source primitive
  误报为 `missing`。
- 非 formal 开发路径仍保留旧 fallback 兼容行为。

## P0-8：compiler fingerprint / model bundle

状态：**已修复并测试**。

- 新增 `model_manifest.json` bundle：
  compiler fingerprint、Region IR schema、calibration hashes、source manifest
  hashes、formal split hash、compatibility policy。
- fit 拒绝混合非 exact fingerprint 的 calibration sources。
- bundle 分开记录 `calibration_source_fingerprint` 与
  `model_builder_fingerprint`：前者保证所有硬件/control artifact 来自同一环境，
  后者保证 fit/evaluate 使用完全相同的模型代码。这样 trace/evaluation 修复不会错误地
  要求重编所有硬件 control，同时也不会静默用不同 builder 消费 bundle。
- evaluate 在加载前校验文件 hash，并与 model builder 的
  compiler/hardware/repository fingerprint 做 exact compatibility gate。

## 验证进度

- focused P0 regression：`37 passed`。
- 全 NKI suite：`297 passed, 65 deselected`，仅有 3 个既有数值 warning。
- `collect/fit/evaluate --dry-run` 已通过真实 parser contract。
- Inf2 实机 smoke：`relu rows=128, F=128, float32` compile/profile/trace 成功。
- full collect 曾启动并完整完成 24 个 Level-B engine controls；该次运行同时暴露并
  修复了 `runtime_overhead.empty` 会使 collect 必然失败的问题。移除该 case 后，
  runtime controls 为 120 个硬件 compile/profile case，整套 canonical
  collect 远大于最初估计，已停止该次非 resume-safe 长跑，避免在本轮内留下一个
  被误认为完整的 MAPE。完整正式实验仍应从空 root 连续运行三阶段命令，只有
  `evaluation/report.json` 通过 35 点断言后才可报告新的 headline。
- 修复 resume 后又从空 root 启动正式 collect：
  `/tmp/nki_cost_model_final`。已完成 24/24 Level-B controls，并完成 37 个
  runtime case（第 38 个开始时按本轮执行时间边界停止）。由于现在命令可 resume，
  可直接重跑同一 collect 命令继续；本轮没有把部分结果计算成 MAPE。

---

# 十八、2026-08-11 P0 最终验收结果

本节是 P0-1 至 P0-8 的最终验收记录，取代上一节中“采集进行中”的临时状态。
正式 root：`/tmp/nki_cost_model_final`。

## 18.1 环境与数据边界

- AWS Inf2：`inf2.xlarge`，2 个 NeuronCore；本实验为单 NeuronCore workload。
- `neuronx-cc`：`2.26.6360.0+6f180f47`。
- calibration 与 holdout 目录严格分离：fit 只读 `microbench/` 与 `controls/`，
  evaluate 才读取 `holdouts/`。
- 未删除任何高误差点；未使用 holdout 拟合；未调 Level-B constants。
- formal split 来自 `microbench/inf2_nki/configs/formal_holdouts.json`，
  evaluate 的 `expected == actual == 35` 强断言通过。

## 18.2 完整采集与 fit

成功完成的 controls：

| Suite | 成功/总数 |
|---|---:|
| Level-B engine lowering | 24/24 |
| Runtime overhead | 120/120 |
| Directional dtype DMA canary | 56/56 |
| FP32 DMA write partition surface | 80/80 |
| BF16 DMA write steady surface | 14/14 |
| Strided DMA surface | 8/8 |
| Structured region controls | 44/44 |

Holdout：

| Split | 成功/总数 |
|---|---:|
| Formal FP32 v1 | 35/35 |
| Auxiliary BF16 v1 | 4/4 |

Fit 输出：

- compute calibration：6 行；
- structured control calibration：124 点；
- structural Static DMA：44 点；
- runtime controls：120 点，fit RMSE `1520.517 ns`；
- strided DMA：8 点；
- bundle：`/tmp/nki_cost_model_final/calibration/model_manifest.json`。

## 18.3 Formal FP32 headline

最终 report：`/tmp/nki_cost_model_final/evaluation/report.json`。

- **Formal FP32 cases：35/35**。
- **Final NC-p50 MAPE：14.9794749449%（四舍五入 14.979%）**。
- compute-only MAPE：`63.2017072216%`。
- compute + DMA MAPE：`31.2963651039%`。
- scheduler resource-overlap MAPE：`32.0058737858%`。
- formal replay 的 compute calibration audit：所有实际 costed compute events 为
  `exact`；纯 DMA interleave 为 `not_applicable`，没有 dtype fallback 或 missing。

分算子 NC-p50 MAPE：

| Operator | Cases | MAPE |
|---|---:|---:|
| interleave | 5 | 8.9377659031% |
| kl_divergence | 4 | 15.2411482656% |
| layernorm | 5 | 30.5833596511% |
| mul2 | 4 | 6.2565309620% |
| relu | 4 | 9.1172135634% |
| rmsnorm | 5 | 22.3972879950% |
| sigmoid | 4 | 8.4042833490% |
| softmax | 4 | 14.6532126915% |

最大单点绝对误差：

- case：`layernorm__r128__c128__float32`；
- signed error：`-38.9407870211%`；
- absolute error：`38.9407870211%`；
- predicted NC：`32.9719750086 us`；
- measured NC p50：`54.0 us`。

Auxiliary BF16（不进入 formal FP32 headline）：

- 4/4 cases；
- NC-p50 MAPE：`33.6208822868%`；
- strict calibration audit 全部为 `exact`。

## 18.4 验收过程中额外修复的 P0 blocker

正式三阶段执行还暴露并修复了以下问题：

1. compiler-invalid `runtime_overhead.empty` 会使 collect 必然返回非零；已从
   canonical runtime matrix 移除。
2. `--skip-existing` 的 case 被 harness 错计为失败，导致 resume 在第一个完整 suite
   后停止；现已计为成功并有回归测试。
3. interleave 的十万级 exact byte segments 会膨胀 JSONL 并让 interval history
   二次退化；超过 1024 segment 时回退 compact conservative range，小型精确 segment
   correctness 仍保留。F=512 smoke：trace 约 3 KB、scheduler 约 0.1 ms。
4. predicate source event 的 `output_dtype=bool` 不能作为 Level-B value dtype；现从
   non-bool input / Region IR 恢复 FP32/BF16 value dtype，strict dtype 仍不允许跨 dtype。
5. provenance bundle 分离 hardware/control source fingerprint 与 model-builder
   fingerprint，既禁止混合硬件/compiler artifact，也要求 fit/evaluate 模型代码 exact。
6. calibration audit 改为检查实际 lowered events，不再误报 raw source primitive。

## 18.5 P0 验收结论

P0-1 至 P0-8 的代码路径、强断言、strict dtype、provenance gate、range/scheduler
correctness 和可 resume 三阶段 pipeline 均已完成。当前可复现的正式 headline 为：

> **35-point formal FP32 NC-p50 MAPE = 14.979%**，最大单点绝对误差
> **38.941%**（`layernorm`, F=128）。
