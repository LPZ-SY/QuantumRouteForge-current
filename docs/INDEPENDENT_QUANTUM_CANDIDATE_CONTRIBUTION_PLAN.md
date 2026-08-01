# 量子候选独立正向贡献：后续确认性实验计划

## 0. 分支与执行边界

本计划仅在分支 `research/independent-quantum-candidate-contribution` 上实施。

- 禁止直接修改、提交或推送 `main`。
- 不覆盖现有 Baihua DeepBlock 竞赛包、历史真机 counts、报告或结论。
- 每个阶段测试通过后单独 commit，并 push 到该研究分支。
- 最终只创建该分支到 `main` 的 Pull Request，由人工审查后决定是否合并；Codex 不得自动合并。
- 禁止 force push、改写历史标签、把 replay/simulator/fallback 结果标成新鲜真机结果。

## 1. 当前问题与研究目标

当前仓库已经证明：Baihua 真机 counts 可以作为局部候选源，在经典容量修复、真实路线距离评价和严格接受规则下产生路线精化。但是现有正式结论没有证明 Baihua 候选源优于相同预算的均匀随机候选源，因此只能称为“硬件辅助正向收益”，不能称为“量子候选的独立正向贡献”。

本轮研究的唯一主问题是：

> 在冻结的局部 QUBO、线路、shots 和评价口径下，Baihua 真机分布是否相对于均匀随机分布，在预先定义的低能候选区域中放置了更多概率质量？

只有该问题通过预注册的确认性检验，才允许表述为：

> 在本次冻结协议与测试范围内，观察到量子候选源相对于均匀随机采样的独立正向采样贡献。

该表述仍不等于量子加速、量子优势、优于经典精确算法或优于所有经典采样器。

## 2. 对当前代码的审计结论

现有实现已经具备本研究的大部分基础：

1. `candidate_evaluator.py` 会读取完整 counts，按频率形成 Top-k 候选，逐个执行容量修复和真实路线距离评价；当前不是只使用最高频 bitstring。
2. `deepblock_solver.py` 使用边界客户、8 比特重叠块、Baihua 线路、候选预算和经典严格接受规则完成闭环。
3. `analyze_baihua_candidate_budget.py` 已能够用历史 counts 研究 k=1、2、4、8、16、32、64 等候选预算。
4. `analyze_quantum_candidate_quality.py` 已提出独立候选质量事件：每个局部 QUBO 中精确能量排名最低的 10% bitstring，并比较 hardware probability mass 与 exact uniform mass。
5. 现有独立候选质量分析属于探索性分析，因为该门槛是在历史硬件数据产生后才引入，不能直接作为确认性证明。

因此下一阶段不应重新发明指标，而应把已经提出的指标正式版本化、预注册，并使用从未参与筛选、调参或报告的新硬件数据进行确认。

## 3. 核心方法：将候选质量与后处理完全解耦

### 3.1 主事件定义

对宽度为 `w` 的冻结局部 QUBO，枚举全部 `2^w` 个 bitstring，按以下键稳定排序：

```text
(proxy_energy, bitstring)
```

定义：

```text
m = ceil(0.10 × 2^w)
L = 排名前 m 的 bitstring 集合
```

主指标为硬件 counts 落入 `L` 的概率质量：

```text
hardware_low_energy_mass = sum(count[z] for z in L) / shots_received
uniform_low_energy_mass = |L| / 2^w
Delta_low = hardware_low_energy_mass - uniform_low_energy_mass
```

当 `w=8` 时：

```text
m = 26
uniform_low_energy_mass = 26 / 256 = 0.1015625
```

这个指标不使用容量修复、路线构造、Top-k、OR-Tools 或最终接受规则，因此能够单独评价量子采样分布本身。

### 3.2 统计单位

- shot 不是独立实验单位。
- block 不是最终确认性统计单位。
- 独立 seed 是确认性统计单位。

同一 seed 的三个冻结 block 和两次硬件重复先在 seed 内求平均，再在 seed 之间做统计，避免伪重复。

### 3.3 主确认性判定

保持现有探索脚本已经提出的冻结规则：

1. seed 级 `Delta_low` 的均值为正；
2. seed bootstrap 95% 置信区间下界大于 0；
3. seed 级精确配对 sign-flip 双侧检验 `p < 0.05`。

另外报告但不替代主判定的实际意义门槛：

- 平均 `Delta_low >= 0.02`，即至少增加 2 个百分点；
- 至少 70% 的独立 seed 为正。

主统计门槛通过，允许称为“独立正向采样贡献”；主统计和实际意义门槛同时通过，才允许称为“稳定且具有实际幅度的独立正向贡献”。

## 4. 消除当前实验中的主要偏差

### 4.1 禁止使用历史筛选 seed

现有报告、试点、深度扫描、候选预算分析或 headroom 筛选使用过的 seed 全部属于开发集，不得进入确认集。

确认集预注册为：

```text
正式 seeds：3001–3016，共 16 个
技术备用 seeds：3017–3024，按顺序使用
smoke seed：2999，仅用于管线检查，不进入正式统计
```

备用 seed 只能在生成硬件任务之前，因为确定性的技术原因替换正式 seed，例如无法生成三个 8 比特 block。不得根据模拟器、精确解、随机结果或硬件结果替换 seed。

### 4.2 不按 headroom 筛选

确认性主指标不需要路线优化余量，因此不得先筛选“容易改善”的实例。所有预注册 seed 都进入主分析。

可以在次要分析中按 exact headroom 分层，但不得影响主样本纳入。

### 4.3 在读取硬件 counts 前冻结子问题

当前闭环会接受前一 block 的候选并改变后续状态。为评价独立采样贡献，确认性主实验必须使用静态冻结子问题：

```text
生成实例
→ 容量约束 K-Means
→ 选定车辆对和边界池
→ 构造 B1、B2、B3 三个唯一 block
→ 冻结 assignments_before、proxy、QASM、参数和客户顺序
→ 最后才允许提交硬件
```

确认性主实验不使用 `B1→B2→B3→B3→B2→B1` 的自适应更新轨迹，也不因任何硬件结果改变后续 block。

## 5. 正式硬件矩阵

推荐冻结矩阵：

```text
16 个独立 seed
× 3 个静态 block
× 2 次硬件重复
= 96 个 Baihua 真机任务
```

固定参数：

```text
customers = 40
vehicles = 4
capacity = 按当前 1.15 平衡容量规则推断
pool_size = 16
block_size = 8
blocks = B1, B2, B3，各执行一次
qaoa_depth = 1
shots = 4096
candidate_k = 64（仅次要 Top-k 分析使用）
overlap = 3
backend = Baihua
filter_extremes = false
routing_method = 与当前正式包一致
```

预计硬件资源：

```text
96 tasks
393,216 shots
```

如果硬件额度不足，不得在看到结果后临时减少任务。应在提交前另建 `protocol-lite-v1`，重新冻结样本量和结论边界。

## 6. 线路、映射与执行顺序

- 整个确认批次固定逻辑 QUBO 构造、QAOA 深度、参数预训练方法和编译选项。
- 在正式批次前读取一次 Baihua calibration，确定并冻结一个满足约束的 8 比特物理子图；后续任务优先使用同一物理 qubit 列表。
- 每个任务保存 logical QASM、physical QASM、物理映射、CNOT、SWAP、深度、两比特层数、calibration 时间和编译通过状态。
- 两次重复必须使用相同逻辑 QASM 和参数，但拥有不同 task ID。
- 提交顺序使用预先固定的随机排列，交错 seed、block 和 repeat，降低时间漂移与某个 seed 绑定。
- 任何中途统计不得改变提交顺序、样本量或线路参数。

## 7. 公平对照

### 7.1 主随机基线

主事件使用 exact uniform mass，不需要蒙特卡洛随机样本：

```text
uniform_low_energy_mass = |L| / 2^w
```

因此不会因为随机基线自身抽样波动而造成结论不稳定。

### 7.2 Top-k 路线贡献对照

次要分析比较相同 4096 shots 和相同 `candidate_k=64` 下：

- Baihua hardware counts；
- uniform multinomial counts；
- ideal simulator counts；
- exact enumeration，仅作为上界。

对每个硬件子问题生成固定随机种子下的多次 uniform multinomial 重采样，完整复用：

```text
频率排序
→ Top-64
→ 容量修复
→ 真实路线距离
→ 严格改善判定
```

不得将硬件 Top-64 与随机“直接任选 64 个唯一状态”混为同一基线。

### 7.3 经典算法边界

即使主检验通过，也只能证明 hardware distribution 相对 uniform 的富集。模拟 QAOA、经典 SA、精确枚举和 OR-Tools 结果均单独报告，不得把“优于均匀随机”改写成“优于经典算法”。

## 8. 指标层级

### 8.1 唯一主指标

```text
seed-level mean Delta_low
```

只有该指标决定是否形成“量子候选独立正向贡献”结论。

### 8.2 关键次要指标

1. hardware mass in route-improving states；
2. hardware mass in low-energy AND route-improving states；
3. Top-64 improving-block hit rate；
4. Top-64 best route improvement；
5. hardware versus uniform Top-64 配对差；
6. hardware versus ideal simulator 的低能质量差；
7. distribution entropy、unique candidate count、top-k probability mass；
8. 结果按 repeat、block、calibration 时段分层的稳定性。

### 8.3 敏感性分析

`k = 1, 2, 4, 8, 16, 32, 64, 128, 256` 只用于候选预算曲线，不改变主门槛。除预注册的 `k=64` 外，其余 k 不得用于选择性主张。

## 9. 失败、重试与排除规则

任务只有满足以下条件才可评估：

- 来源为新鲜 Baihua hardware；
- task ID 唯一；
- counts 非空且总和等于 4096；
- bitstring 宽度与冻结 block 一致；
- requested backend 与 actual backend 均为 Baihua；
- protocol/config/proxy/QASM/customer-order hashes 一致；
- 编译审计通过预设 CNOT、depth 和 mapping 约束。

重试规则：

- 网络、队列或结果获取失败时，允许对同一冻结任务最多重试 2 次；
- 重试不改变 QUBO、QASM、参数、shots 或物理 qubit 列表；
- 只有最后一个成功 task 进入分析，全部 task ID 和失败原因仍保留；
- 不允许用 simulator、local exact 或 fallback 替代硬件结果；
- 持续失败的 seed 按预注册规则处理，不得查看候选质量后决定排除。

## 10. 防止后验调整

正式硬件提交前必须生成并提交：

```text
docs/INDEPENDENT_QUANTUM_CANDIDATE_CONTRIBUTION_PROTOCOL.md
experiments/configs/independent_candidate_contribution_v1.json
experiments/manifests/independent_candidate_contribution_v1.json
experiments/manifests/independent_candidate_contribution_v1.sha256
```

manifest 至少冻结：

- seed 和备用 seed；
- instance、capacity 和客户数据哈希；
- assignments_before；
- block 客户顺序；
- proxy 全部系数；
- low-energy 集合及其哈希；
- QAOA 参数；
- logical QASM 哈希；
- backend、shots、repeat；
- 物理 qubit 列表和编译限制；
- 主判定规则、统计单位、排除和重试规则；
- 当前 Git commit 和依赖快照。

提交后，代码应拒绝在工作树不干净、commit/tag 不匹配或 manifest hash 不匹配时运行正式真机任务。

## 11. 代码实施阶段

### R0：冻结研究分支基线

- 确认当前分支；
- 运行完整 pytest；
- 记录 main 基线 commit；
- 创建研究审计文档；
- 不改现有报告结论。

### R1：协议与配置模型

创建版本化 JSON schema，明确主指标、seed、矩阵、重复、失败规则和结论语言。

### R2：静态子问题 manifest

新增离线脚本，生成 16 个正式 seed 的 B1/B2/B3，验证每个 block、低能集合、QASM 和哈希；该阶段禁止访问硬件。

### R3：可恢复硬件批处理

实现 one-task cap、显式双确认、断点恢复、幂等 task key、原始 counts 归档、失败重试和来源隔离。

### R4：确认性分析器

在现有探索代码基础上新增 confirmatory 模式，严格读取冻结 manifest；计算 block、repeat、seed 三层结果，但仅以 seed 作为确认统计单位。

### R5：公平 Top-k 后选择对照

实现 uniform multinomial、ideal simulator 和 hardware 的相同 shots、相同 Top-k、相同修复与真实路线评价。

### R6：测试与 CI

覆盖排名 ties、bit order、uniform exact mass、seed 聚合、重复平均、manifest 防篡改、任务幂等、失败重试和来源隔离。

### R7：离线 dry-run 与 smoke

- 使用开发 seed 和模拟/历史 replay 完成全流程；
- 使用 seed 2999 运行最多 3 个新鲜硬件 smoke 任务；
- smoke 只验证管线，不进入统计；
- smoke 后不得根据质量结果修改主指标。

### R8：正式 96 任务执行

- 先输出完整任务表和预计 shots；
- 用户逐批确认；
- 默认每次最多提交 1 个任务；
- 不显示或使用中期主统计做决策。

### R9：最终分析与 PR

- 完整性验证通过后一次性解锁确认性分析；
- 生成报告、CSV、JSON、图表和结论边界；
- 测试与 CI 通过后创建 PR，不自动合并。

## 12. 建议新增文件

```text
experiments/configs/independent_candidate_contribution_v1.json
experiments/prepare_independent_candidate_manifest.py
experiments/run_independent_candidate_batch.py
experiments/validate_independent_candidate_store.py
experiments/analyze_independent_candidate_contribution.py
experiments/replay_independent_candidate_counts.py
experiments/uniform_topk_counterfactual.py
docs/INDEPENDENT_QUANTUM_CANDIDATE_CONTRIBUTION_PROTOCOL.md
docs/independent_candidate_contribution_report.md
tests/test_independent_candidate_protocol.py
tests/test_independent_candidate_batch.py
tests/test_independent_candidate_analysis.py
```

优先复用 `candidate_evaluator.py`、`deepblock_builder.py`、`sparse_proxy_qubo.py`、`qaoa_depth_runner.py` 和现有 logger。不要复制一套不一致的 bitstring、修复或路线评价逻辑。

## 13. 结果证据归档

当前 `.gitignore` 排除了 `results/`。正式证据需要：

1. 本地保存完整原始 counts、QASM、映射、校准、task ID 和 replay record；
2. 生成不可变 ZIP 和逐文件 SHA-256 manifest；
3. 在 Git 仓库中提交配置、协议、任务清单、哈希清单、聚合 CSV/JSON 和报告；
4. 大型原始 evidence 使用 GitHub Release、Actions artifact 或其他只读归档保存，报告中记录下载位置和总哈希；
5. 不得只提交汇总数字而无法回放原始 counts。

## 14. 最终结论模板

### 主检验通过、路线次要指标未通过

> 在冻结的局部 QUBO 候选定义、新 holdout seeds 和 Baihua 真机批次中，硬件分布在预先定义的低能区域中相对于均匀随机分布表现出显著概率质量富集，支持量子候选源具有独立正向采样贡献。该贡献尚未稳定转化为最终路线距离改善，不构成量子加速或对经典算法的优势证明。

### 主检验和路线次要指标均通过

> 在本次冻结协议范围内，Baihua 真机分布不仅在预定义低能区域中相对于均匀随机显著富集，而且在相同 shots、Top-k 和经典后处理预算下提高了局部改善候选的命中率。结果支持候选层与下游局部精化层的独立正向量子贡献，但不外推为普遍量子优势或速度优势。

### 主检验未通过

> 新确认批次未能证明 Baihua 真机分布相对于均匀随机在预定义低能候选区域中具有稳定富集。现有结果仍只能支持硬件候选可以进入经典闭环并偶尔产生改善，不能升级为量子候选的独立正向贡献。

## 15. Codex 汇报要求

每个阶段必须汇报：

```text
阶段与状态
当前分支和 commit
修改文件
执行命令
测试结果
是否访问硬件
新 task ID、backend 和 shots（如有）
生成产物及 SHA-256
风险、阻塞和下一步
```

禁止仅报告“已完成”，禁止只展示表现最好的 seed，禁止把探索性敏感性结果替代预注册主指标。
