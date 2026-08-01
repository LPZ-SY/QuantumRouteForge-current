# 摘要

容量约束车辆路径问题（Capacitated Vehicle Routing Problem，CVRP）是物流配送中的典型组合优化问题。现阶段含噪中等规模量子（NISQ）处理器难以直接承载完整CVRP编码，而小规模量子演示又容易落入“量子线路运行了，但量子候选是否优于随机并不清楚”的证据缺口。本文提出DeepBlock拓扑协同混合精化框架：经典端先产生满足容量约束的初始车辆分配，再从相邻车辆的边界客户中构造三个最多8变量、重叠3客户的局部块；量子端将每个局部重分配问题编码为与Baihua物理链直接对齐的稀疏QUBO，并以QAOA生成候选；经典端负责容量修复、真实路线距离计算和单调接受。正式实验包含6个预先确认存在局部精化余量的40客户、4车辆实例，共执行36个Baihua真机任务，每任务4096 shots，总计147,456 shots，全部实现0 SWAP映射，并完成144/144条全臂离线重放。

为独立界定量子采样的正向贡献，本文规定每个8比特QUBO完整状态空间中能量最低10%的状态为“高质量量子候选”，该阈值仅由QUBO决定，不读取真机counts，也不依赖后续路线接受结果。Baihua在该区域的平均概率质量为27.76%，均匀随机为10.16%，理想QAOA为39.18%；真机相对随机富集2.73倍，差值为+17.60个百分点，6/6个独立种子方向一致，精确配对符号翻转检验p=0.03125。端到端层面，Baihua在6/6个实例中产生严格路线改善，平均改进2.555，恢复71.4%的局部精确余量；随机、理想模拟和局部精确枚举的平均改进分别为2.757、2.888和3.653。Baihua与随机为3胜0平3负，差异不显著（p=0.65625）。实验进一步表明，将在线候选预算由k=8扩展到k=64，可使种子2/3/4的真机平均改进由1.593提高至3.475；相较于继续加深QAOA，改善解码预算更能释放硬件样本价值。

本文的结论边界是：Baihua真机显著提高了低BQM能量候选的采样质量，并在可审计闭环中产生了有效路线精化；但现有证据不支持相对于经典随机、OR-Tools或精确枚举的量子加速。该研究为量子优化竞赛提供了一套可复现、可否证且避免过度宣称的真机评测范式。

**关键词：** 容量约束车辆路径问题；量子近似优化算法；QUBO；混合量子-经典优化；真机采样；候选质量；可复现评测

# Abstract

The Capacitated Vehicle Routing Problem (CVRP) is a representative combinatorial optimization problem in logistics. Current noisy intermediate-scale quantum processors cannot directly host full-scale CVRP encodings, while small hardware demonstrations often leave a critical evidentiary gap: a quantum circuit may run successfully without showing that its candidates are better than uniform random samples. This paper presents DeepBlock, a topology-co-designed hybrid refinement framework. A classical stage first constructs a capacity-feasible assignment and selects boundary customers from a pair of neighboring routes. Three overlapping local blocks, each containing at most eight binary decisions, are then mapped to sparse QUBOs aligned with a calibrated physical chain on the Baihua superconducting processor. QAOA generates reassignment candidates, whereas capacity repair, true route-distance evaluation, and monotone acceptance remain classical.

The formal study uses six 40-customer, four-vehicle instances with pre-identified local refinement headroom. We execute 36 Baihua hardware jobs at 4,096 shots each, totaling 147,456 hardware shots. All formal circuits are mapped with zero SWAP overhead, and 144/144 replay records across the exact, random, ideal-simulator, and hardware arms are verified. To isolate quantum candidate quality, we define the exact bottom 10% of QUBO energies within each eight-bit state space as the high-quality region. This threshold is determined from the QUBO alone, before hardware counts or downstream route decisions are read. Baihua assigns 27.76% probability mass to this region, compared with 10.16% under uniform sampling and 39.18% under ideal QAOA. The hardware enrichment is 2.73x, is positive for all six independent seeds, and yields an exact paired sign-flip p-value of 0.03125.

At the end-to-end level, the hardware arm improves all six instances, with a mean route-distance reduction of 2.555 and a mean recovery of 71.4% of exact local headroom. The matched random, ideal-simulator, and exact-enumeration arms achieve mean improvements of 2.757, 2.888, and 3.653, respectively. The hardware arm records three wins and three losses against random sampling, with no significant paired difference. These results establish a positive hardware contribution to low-energy candidate generation and a functional hybrid refinement loop, but do not establish quantum speedup or superiority over classical optimization.

**Keywords:** capacitated vehicle routing; QAOA; QUBO; hybrid quantum-classical optimization; quantum hardware; candidate quality; reproducible benchmarking

[[PAGEBREAK]]

# 1 引言

车辆路径问题源于Dantzig和Ramser对油品配送调度的研究[1]，此后形成了包括容量约束、时间窗、多仓库和异构车队等在内的庞大模型体系[2]。CVRP要求每位客户恰好被一辆车服务，每条路线从仓库出发并返回仓库，且车辆载重不超过容量上限。该问题兼具分配和排序两类离散决策，随着客户数量增长，其解空间迅速膨胀。经典运筹优化、元启发式和工业求解器仍是实际应用的主力，但量子计算为组合优化提供了新的采样与启发式搜索机制。

QUBO及其等价Ising形式为经典组合问题映射到量子优化算法提供了统一接口[4,5]。QAOA通过交替施加问题哈密顿量和混合哈密顿量，在参数化量子态中提升低能状态的测量概率[6]。然而，NISQ设备受量子位数量、连接拓扑、门错误、退相干和测量噪声限制[3]。对CVRP而言，直接位置编码通常需要随客户数二次增长的量子位，硬约束还会引入额外松弛变量和罚项[13-15]。因此，现阶段更现实的路线不是让量子机替代完整经典求解器，而是将量子处理器嵌入局部精化、列生成或大邻域搜索等混合框架。

现有量子车辆路径研究已经覆盖量子退火[11]、QAOA/变分算法[12-14]和约束保持混合器[15]，但常见局限包括：问题规模很小、主要依赖模拟器、缺少同预算随机基线、只报告最优样本而不分析分布质量，或将经典后处理带来的改善归因于量子部分。对于竞赛项目，这些缺口会直接削弱“量子真正起到什么作用”的说服力。

本文围绕三个研究问题展开：

1. **RQ1：量子候选质量。** 在不使用最终路线接受结果的前提下，Baihua真机是否显著提高低BQM能量候选的概率质量？
2. **RQ2：端到端有效性。** 真机候选进入容量修复和真实路线评价后，能否稳定产生CVRP局部精化？其质量与同预算随机、理想模拟和局部精确枚举相比如何？
3. **RQ3：工程瓶颈。** 当前效果主要受QAOA深度、硬件噪声、局部块结构还是候选解码预算限制？

本文的主要贡献如下。

- 提出DeepBlock局部精化框架，以16客户边界池构造三个最多8变量、重叠3客户的局部块，并采用正向-反向扫描形成逐块闭环。
- 提出拓扑协同稀疏QUBO，使保留的二次项仅落在选定物理链的直接耦合边上；正式Baihua任务全部0 SWAP。
- 提出独立量子候选质量门槛：以每个QUBO完整状态空间的最低10%能量态定义高质量候选，避免用最终路线改善反推量子贡献。
- 建立包含随机、理想模拟、精确枚举和真机四臂的配对实验，公开无精化余量比例、失败深度和不显著结果。
- 保存counts、逻辑与物理QASM、物理映射、校准时间、任务ID、候选评价和状态转移，形成可离线重放的证据链。

[[FIGURE:fig0_method_overview.png|图1 DeepBlock拓扑协同真机混合精化流程。量子处理器只负责候选生成；约束修复、真实路线评价和接受决策均在经典端完成。]]

# 2 相关工作

## 2.1 QUBO、Ising与QAOA

QUBO以二进制变量的线性项与二次项表示离散目标，可通过变量变换映射为Ising哈密顿量[4,5]。QAOA最初由Farhi等提出[6]，深度p控制问题相位算子与混合算子的交替层数。后续研究将其推广为Quantum Alternating Operator Ansatz，以构造保持可行子空间的混合器[7]；Zhou等系统讨论了参数初始化、深度、机制和近端设备实现[8]；变分量子算法综述则指出，硬件噪声、优化景观和测量成本共同限制了近期应用[9]。

对NISQ实验而言，较深线路并不自动带来更好结果。增加p虽可提升理想态表达能力，却也线性增加双量子位门层数，并放大校准漂移与退相干影响。因此，线路结构、物理映射、参数迁移和下游解码预算必须联合设计，而不能只比较p的大小。

## 2.2 量子车辆路径优化

Feld等采用量子退火器构造CVRP混合求解流程[11]；Harwood等总结了在量子计算机上建模和求解路径问题的关键编码与资源挑战[12]。Azad等研究了QAOA求解小规模VRP，指出结果对实例、优化器、参数初值和深度高度敏感[13]。Fitzek等将异构车队路径问题映射为Ising模型，并在最多21量子位的模拟中展示量子位需求随客户数二次增长[14]。Xie等通过约束保持混合器提升CVRP可行解和最优解测量概率[15]。近期也有将QAOAnsatz嵌入列生成子问题的工作[17]。

上述研究支持“分解+量子子问题+经典主问题”的总体方向。本文与其区别在于：第一，直接在Baihua真机上运行逐块闭环，而不是仅做模拟；第二，将物理链作为QUBO稀疏化约束；第三，把低能候选概率质量作为独立量子贡献指标；第四，明确将端到端路线质量与量子候选质量分层报告。

## 2.3 Quafu真机与可审计实验

Quafu-Qcover提供QUBO、QAOA、图分解和云端硬件执行的一体化流程，并支持通过任务ID回收结果[10]。本文使用同一云平台生态中的Baihua超导处理器。与只保存最终路线不同，本文同时固化校准快照、逻辑线路、物理线路、物理量子位链、counts和每个子问题的状态前后快照，以区分“线路提交成功”“返回有效采样”和“采样真正改变优化状态”三个层次。

# 3 问题定义与评价边界

## 3.1 CVRP定义

给定仓库节点0、客户集合V={1,...,n}、车辆集合K={1,...,m}、客户需求d_i、车辆容量Q以及欧氏距离c_ij，目标是寻找m条从仓库出发并返回仓库的路线，使每个客户恰好被访问一次、每条路线负载不超过Q，并最小化总路线距离：

[[EQUATION:min  Σ(k∈K) Σ(i,j∈V∪{0}) c_ij y_ijk]]

其中y_ijk表示车辆k是否从节点i行驶到节点j。完整CVRP包含客户分配、访问顺序、流守恒、容量和子回路消除等约束。本文不把完整模型直接映射到量子位，而是固定大部分当前解，仅允许一个边界客户块在两条车辆路线之间重新分配。

## 3.2 局部量子决策

对选定车辆对(a,b)和局部客户块B={v_1,...,v_w}，w≤8，为每位客户定义一个二进制变量x_i：x_i=0表示分配给车辆a，x_i=1表示分配给车辆b。块外客户保持不变。局部代理QUBO写为：

[[EQUATION:E_Q(x)=c+Σ(i=1..w) h_i x_i+Σ(i<j) J_ij x_i x_j]]

线性项近似客户移动对边界几何与车辆负载的影响，二次项表示客户共同移动时的相互作用。为减少路由开销，只保留选定物理子图支持的逻辑二次项；其余项记录为被剪枝交互，供审计和后续代理改进使用。

## 3.3 结论边界

本文区分三类主张：

- **线路有效性：** 任务确实提交至Baihua，返回4096 shots的counts，并通过映射与重放检查。
- **量子候选质量：** 真机对预设低BQM能量区域的概率质量高于同一QUBO上的均匀随机分布。
- **端到端优化优势：** 真机最终路线质量显著优于匹配的经典基线。

本文支持前两项中的“线路有效性”和“量子候选质量正向贡献”，并支持“端到端路线得到改善”；但不支持相对于随机或工业经典求解器的端到端量子优势。

# 4 DeepBlock拓扑协同混合方法

## 4.1 初始解与边界客户池

每个实例首先使用容量约束聚类获得4辆车的可行客户分配，再对每条路线采用最近邻和两轮2-opt形成初始路线。算法按车辆路线间几何邻近度选择一对车辆，从两条路线边界区域选出最多16位客户。候选分数综合客户到两条路线的相对距离、局部邻域和负载平衡信息。

## 4.2 重叠局部块

边界池被构造成最多三个8客户块B1、B2和B3，相邻块默认重叠3位客户。算法使用B1→B2→B3→B3→B2→B1的双向扫描。重叠使前一块接受的客户重新分配能够在后续块中被再次协调，反向扫描则用于消解单向顺序偏差。若边界池不足8位，块宽自然退化，不引入虚拟客户或无效量子位。

## 4.3 物理拓扑选择与QUBO稀疏化

算法从Baihua校准快照中选择一条8量子位高保真物理链。正式实验使用物理量子位[41,28,29,30,17,16,15,14]，对应7条直接耦合。QUBO仅保留映射到这些直接耦合的J_ij，从建模阶段避免后续SWAP。所有被剪枝的二次项、保留原因和物理耦合保真度均写入日志。

## 4.4 QAOA线路与参数

对深度p，QAOA状态写为：

[[EQUATION:|γ,β⟩=Π(l=1..p) exp(-iβ_l H_M) exp(-iγ_l H_C)|+⟩^⊗w]]

其中H_C由稀疏QUBO映射得到，H_M为X混合哈密顿量。参数先在理想状态向量上进行确定性坐标搜索，并在相邻块间迁移。正式配置选择p=1；p=2用于深度对照，p=3因预设门控未通过而停止真机扩展。逻辑线路编译到Baihua原生门集后，正式任务平均电路深度为27.7，平均双量子位门层数为13.3。

## 4.5 候选解码与经典接受

真机返回counts后，按测量频次排序bitstring。在线配置最多对前k=64个候选进行完整目标评价：将bitstring解码为车辆分配，执行确定性容量修复，重建路线并计算真实总距离。只有候选距离严格小于当前距离时才接受；否则保持当前解。接受后的分配立即用于构造下一块QUBO，因此36个正式真机任务是逐块闭环而非一次性静态批处理。

诊断代码可在实验结束后离线计算全部已采样状态的真实距离，用于best-of-shots和候选预算曲线；这些离线诊断不改变在线接受结果。论文不将该离线全量评价计入“量子减少经典评价次数”的证据。

## 4.6 独立量子候选质量门槛

为回应“量子部分是否只是可有可无的随机扰动”，本文对每个宽度w=8的QUBO精确枚举2^8=256个能量。按(E_Q(x), bitstring)稳定排序，预先固定前ceil(0.10×256)=26个状态为高质量集合G_0.1。该集合在读取真机counts前即可确定。

定义真机低能概率质量：

[[EQUATION:P_HW(G_0.1)=Σ(x∈G_0.1) count_HW(x)/4096]]

均匀随机基线为|G_0.1|/256=10.15625%。量子采样正向贡献的主判据为：以独立种子为统计单位，P_HW(G_0.1)-P_U(G_0.1)的95%种子bootstrap置信区间下界大于0，且双侧精确符号翻转检验p<0.05。真实路线改善、容量修复和top-k接受均不参与这一判定。

由于该门槛是在现有真机数据采集后根据评审建议引入，本文将现有分析标记为探索性证据；门槛与判定规则现已冻结，可供后续新批次作确认性检验。

## 4.7 算法流程

1. 构造容量可行初始分配与路线。
2. 选择相邻车辆对和16客户边界池。
3. 构造三个重叠8客户DeepBlock。
4. 根据最新校准选择8量子位物理链。
5. 构造只保留物理直接耦合的稀疏QUBO。
6. 预训练QAOA参数、编译并提交Baihua任务。
7. 回收counts，计算独立低能候选质量指标。
8. 在线评价前k个候选，修复容量并计算真实路线距离。
9. 若严格改善则更新当前解，并进入下一局部块。
10. 保存全套审计工件并执行离线重放。

# 5 实验设计

## 5.1 实例与预筛选

实验使用确定性随机种子生成二维欧氏CVRP实例，每个实例包含40位客户、4辆同容量车辆，客户需求取正整数。若未显式指定容量，则按总需求/车辆数的1.15倍平衡系数向上取整。首先在种子1—10上运行局部精确枚举，以识别初始DeepBlock是否存在严格改善空间。种子2、3、4、7、9、10具有局部余量，种子1、5、6、8无局部余量，无余量比例为40%。

正式真机集使用上述6个有余量种子。该选择避免在不存在局部改善的实例上浪费真机额度，但会形成条件样本；因此论文同时报告10种子筛选比例，并不把6种子结果外推为所有CVRP实例的无条件平均性能。

## 5.2 对照臂

| 对照臂 | 候选来源 | shots/状态数 | 下游评价与接受 |
|---|---|---:|---|
| random | 8比特均匀随机采样 | 4096 | 与真机相同的k、修复、路线评价和单调接受 |
| sim | 同一QAOA模型的理想状态向量采样 | 4096 | 与真机相同 |
| baihua | Baihua超导真机counts | 4096 | 与真机闭环相同 |
| exact | 8比特局部空间精确枚举 | 256 | 用于测量局部可用余量，不作为可扩展算法 |

## 5.3 正式配置

| 参数 | 正式值 | 说明 |
|---|---:|---|
| 客户数/车辆数 | 40/4 | 完整CVRP规模 |
| 边界池 | 16 | 从两条相邻车辆路线选择 |
| 局部块宽度 | 8 | 对应8个真实二进制决策 |
| 重叠客户数 | 3 | 相邻DeepBlock共享 |
| 扫描顺序 | 双向 | B1→B2→B3→B3→B2→B1 |
| QAOA深度 | p=1 | 由深度门控选定 |
| 真机shots | 4096 | 每局部块 |
| 在线候选预算 | k=64 | 高频前64个完整评价 |
| 正式种子 | 2,3,4,7,9,10 | 均具有局部精化余量 |

## 5.4 评价指标与统计单位

端到端指标包括：相对初始路线的距离改进、局部精确余量恢复率、接受移动数以及相对随机基线的配对差。量子候选质量指标为最低10%BQM能量区域的概率质量及其相对均匀随机的富集倍数。

主统计单位为独立种子，而非同一种子内部的6个相关子问题或4096个相关shots。均值置信区间使用固定随机种子的20,000次种子bootstrap。配对差异使用全部2^6种符号组合的双侧精确符号翻转检验。由于n=6，若6个种子差值同向，双侧检验可达到的最小p值为0.03125。

# 6 实验结果

## 6.1 真机完整性与可重放性

两个正式批次共包含36个Baihua任务，总计147,456真机shots。36/36任务完成编译检查，36/36使用0 SWAP映射，所有任务返回有效counts。正式配置在exact、random、sim和baihua四臂共形成144条重放记录，144/144通过接受bitstring、best-of-shots和状态转移一致性检查。校准时间覆盖2026-08-01 12:43:01至12:49:12。

| 审计项 | 结果 |
|---|---:|
| Baihua正式任务数 | 36 |
| 每任务shots | 4096 |
| 总真机shots | 147,456 |
| 编译通过 | 36/36 |
| 0 SWAP | 36/36 |
| 平均电路深度 | 27.7 |
| 平均双量子位门层数 | 13.3 |
| 全臂重放通过 | 144/144 |

## 6.2 独立量子候选质量

[[FIGURE:fig_quantum_candidate_quality.png|图2 预设最低10%BQM能量区域的概率质量。Baihua显著高于均匀随机，但仍低于理想QAOA。]]

Baihua在最低10%BQM能量区域的平均概率质量为27.76%，均匀随机为10.16%，理想QAOA为39.18%。真机相对随机的富集倍数为2.73，绝对差为+17.60个百分点；以种子为单位的95% bootstrap区间为[+11.05,+23.26]个百分点，6/6个种子差值为正，双侧精确符号翻转检验p=0.03125。

| 指标 | Baihua | 均匀随机 | 理想QAOA |
|---|---:|---:|---:|
| 最低10%能量区域概率质量 | 27.76% | 10.16% | 39.18% |
| 相对随机富集倍数 | 2.73× | 1.00× | 3.86× |
| 相对随机额外概率质量 | +17.60 pp | 0 | +29.02 pp |

若以“超出随机的额外低能概率质量”为分母，Baihua实现了理想QAOA额外富集能力的约60.6%，即(27.76-10.16)/(39.18-10.16)。这说明真机输出并非近似均匀随机，而是保留了显著的QAOA低能集中趋势。

## 6.3 端到端路线精化

[[FIGURE:fig1_formal_seed_improvements.png|图3 六个正式种子上的路线距离改进。Baihua在6/6实例中产生严格改善，但绝对平均改进未超过随机或理想模拟。]]

| 方法 | 正向实例 | 平均距离改进 | 95% bootstrap区间 | 平均局部精确余量恢复 |
|---|---:|---:|---:|---:|
| Baihua真机 | 6/6 | 2.555 | [1.178,4.606] | 71.4% |
| 均匀随机 | 5/6 | 2.757 | [1.074,4.839] | 65.8% |
| 理想模拟 | 6/6 | 2.888 | [1.466,4.726] | 85.1% |
| 局部精确枚举 | 6/6 | 3.653 | [1.759,5.622] | 100.0% |

逐种子改进如下。

| 种子 | Baihua | 随机 | 理想模拟 | 精确余量 | Baihua-随机 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2.330 | 3.002 | 2.198 | 5.052 | -0.672 |
| 3 | 0.625 | 0.728 | 0.815 | 0.815 | -0.103 |
| 4 | 7.469 | 7.382 | 7.023 | 8.038 | +0.087 |
| 7 | 2.012 | 1.987 | 3.149 | 3.660 | +0.025 |
| 9 | 1.987 | 3.445 | 3.237 | 3.445 | -1.458 |
| 10 | 0.908 | 0.000 | 0.908 | 0.908 | +0.908 |

Baihua相对随机为3胜0平3负，平均改进差为-0.202，双侧精确符号翻转检验p=0.65625。因此不能拒绝“Baihua与随机在端到端路线改进上无差异”的零假设。值得注意的是，按每个实例的精确余量归一化后，Baihua平均恢复率71.4%高于随机的65.8%；但该指标同样受小样本影响，不构成优势主张。

## 6.4 候选预算效应

[[FIGURE:fig3_candidate_budget_hit_rate.png|图4 已有真机counts的候选预算诊断。增加k显著提高发现局部改善状态的子问题比例。该曲线是固定前状态的反事实诊断，不等同于闭环最终路线。]]

在p=1的24个历史硬件子问题中，高频前8个候选仅有3个子问题包含改善解，命中率12.5%；k=16、32和64分别提高到29.2%、33.3%和37.5%；评估全部已采样状态时达到58.3%。p=2的对应命中率从k=8的25.0%提高到k=64的41.7%，全量状态达到70.8%。这表明更深线路确实可能采到更多低频好解，但概率质量未稳定集中到最常见的少数bitstring。

为验证闭环效果，本文在种子2/3/4上将Baihua p=1的在线候选预算从k=8提升至k=64。平均路线改进由1.593提升至3.475，增幅约118%，而量子线路深度和shots均未增加。因此，当前系统的主要限制之一是经典端未充分利用已经付费获得的量子样本，而不是量子线路绝对缺少好候选。

[[FIGURE:fig5_hardware_intervention.png|图5 深度与解码预算干预。相较于从p=1加深到p=2，将p=1的候选预算从k=8提高至k=64在种子2/3/4上更稳定。]]

## 6.5 十种子模拟消融

[[FIGURE:fig4_ablation.png|图6 10个配对种子的模拟消融。p=1、k=64在候选预算组中获得最佳平均改进；p=2未表现出稳定优势。]]

| 配置 | 模拟平均改进 | 随机平均改进 | 主要结论 |
|---|---:|---:|---|
| p1,k8,重叠3,双向 | 1.054 | 0.917 | 原始参考配置 |
| p1,k16,重叠3,双向 | 1.468 | 1.159 | 增加预算有效 |
| p1,k32,重叠3,双向 | 1.321 | 1.520 | 轨迹非单调 |
| p1,k64,重叠3,双向 | 1.733 | 1.654 | 最佳模拟平均改进 |
| p2,k8,重叠3,双向 | 0.675 | 0.917 | 深度增加未转化为收益 |
| p2,k32,重叠3,双向 | 1.407 | 1.520 | 仍未超过匹配随机 |
| p1,k32,无重叠,双向 | 1.187 | 1.460 | 重叠有一定结构价值 |
| p1,k32,重叠3,正向 | 1.226 | 1.270 | 双向扫描略有帮助 |

消融结果支持p=1、k=64、重叠3、双向扫描作为正式配置。需要强调的是，随机基线也会随k变化，因为更大的完整评价预算同样提高随机候选被接受的概率。因此，候选预算带来的收益是混合系统层面的资源配置改进，不应全部归因于量子线路。

## 6.6 深度-噪声权衡

在种子2/3/4的k=8真机对照中，p=1的平均改进为1.593，p=2为1.212。p=2在种子3上恢复约89.3%的精确余量并同时超过随机和理想模拟，但种子2和4表现退化。候选预算诊断又显示p=2全量样本中改善状态更多，说明其表达能力与高频集中度之间存在矛盾：更深QAOA可能将概率分散到更多可用低频状态，但硬件噪声和解码预算阻止其形成稳定端到端收益。

基于预设门控“p=2未稳定优于p=1”，本文未继续提交p=3真机任务。该停止规则避免在缺乏正向趋势时盲目消耗硬件额度。

# 7 讨论

## 7.1 量子部分究竟起到了什么作用

本文最强的量子证据不是“最终路线超过经典”，而是“真机显著富集预设低BQM能量区域”。这一结论与经典接受规则解耦：即使不执行容量修复、不计算真实路线、不选择top-k，Baihua在低能区域的概率质量仍为随机的2.73倍。它证明QAOA相位结构在真实硬件上留下了可测量的优化信号，而不是产生完全均匀的随机bitstring。

与此同时，量子低能质量没有充分转化为CVRP优势。对历史硬件子问题，代理能量与真实路线距离的平均Spearman秩相关接近0（p=1约-0.025，p=2约-0.038）。这说明当前主要科学瓶颈是代理目标对齐，而非“量子线路完全无效”。量子机正在优化其被赋予的QUBO，但该QUBO尚不能稳定排序真实路线候选。

## 7.2 为什么不能宣称量子优势

第一，正式真机的绝对平均路线改进低于随机，且配对差异不显著。第二，每个量子局部块只有8变量，256个状态可被经典轻易枚举，因而不存在计算复杂度优势。第三，所有在线移动都经过经典单调接受，“6/6正向”部分来自不接受更差解的安全机制。第四，诊断阶段计算全部已采样候选的真实路线距离，因此当前实现不能用来证明量子减少了经典目标函数调用。

本文使用“有效真机混合优化”描述系统：量子counts确实进入决策链并改变路线，但经典端仍承担可行性、真实目标和安全接受。该表述比“量子加速”更符合证据。

## 7.3 对量子优化工程的启示

实验显示，NISQ组合优化需要同时设计四个层面：代理目标、物理拓扑、采样分布和经典解码。只提升其中一项未必改善端到端结果。本文的0 SWAP映射减少了不必要的路由门；最低10%能量富集证明线路保留了优化信号；k=64改进证明低频样本具有价值；而近零代理相关性解释了为何这些进步尚未成为CVRP优势。

因此，下一阶段优先级应是：使用真实路线增量或学习型代理重构h_i和J_ij；在训练集上冻结代理，再在独立测试种子上验证能量-真实距离相关性；在不扩大完整目标评价预算的条件下比较量子、重要性随机、模拟退火和经典局部搜索的sample efficiency；最后再扩大到14—20个真实决策量子位。

## 7.4 第二硬件平台证据

项目另有Shenglian宽度扫描，覆盖6—22个真实决策量子位、每宽度8192 shots。相对均匀随机合法样本的分布质量信号在全部宽度为正，top-5精确最优的稳定边界位于14量子位；16—22量子位仍有分布级信号，但精确恢复不稳定。该扫描使用2车辆、seed 2026和链式Max-Cut目标，与本文Baihua 40客户、4车辆局部精化任务不同，因此只能作为第二平台可运行性和宽度扩展证据，不能与Baihua最终路线改进直接比较。

# 8 有效性威胁与局限性

## 8.1 内部有效性

- **阈值后设。** 最低10%BQM能量门槛是在已有数据采集后根据评审意见提出，现有p=0.03125属于探索性结果。门槛已冻结，仍需新批次确认。
- **轨迹依赖。** random、sim和baihua会接受不同移动，后续QUBO也随之不同。端到端比较以独立臂闭环为单位，而不能把后续单块counts视为同一QUBO的简单配对。
- **诊断与在线预算。** 全量best-of-shots只用于离线分析；若在运行时同步计算，会增加经典成本并削弱sample-efficiency主张。
- **硬件批次。** 正式任务主要在同一日期和相近校准窗口执行，无法充分评估跨日漂移。

## 8.2 外部有效性

- 实例为确定性合成二维欧氏CVRP，尚未覆盖道路网络、时间窗、动态订单和异构车辆。
- 正式真机集只包含6个有局部余量种子；10种子总体中40%没有精化空间。
- 每个量子核为8比特，可被经典枚举；结果不能外推到经典不可枚举规模。
- Baihua和Shenglian任务定义不同，不能形成严格跨平台性能排名。

## 8.3 构念有效性

低BQM能量质量衡量“量子是否优化了所给代理目标”，并不等同于“量子是否优化了CVRP真实目标”。本文通过分层报告避免混淆，但也暴露出代理建模需要改进。端到端路线改善则包含量子采样、经典修复、路径构造和接受规则的共同作用，不能被单独归因于量子线路。

## 8.4 统计结论有效性

n=6导致置信区间较宽，双侧精确符号翻转检验的最小p值为0.03125。未来应至少使用30个预注册种子、跨3—5个校准日期重复，并将无余量实例按零改善纳入意向性总体结果。同时应报告效应量和置信区间，而不是只依赖显著性阈值。

# 9 可复现性与开放证据

每个子问题保存以下工件：配置文件、实例数据、逻辑QASM、物理QASM、物理量子位和耦合映射、校准时间、原始counts、QAOA参数、代理QUBO、候选评价、接受前后分配、任务ID和消息状态。离线重放脚本重新构造客户、分配和QUBO，使用相同bitstring规范、容量修复和路线目标验证接受结果。

正式证据包包含：

- `formal_seed_results.csv`：六个种子的四臂结果；
- `formal_summary.json`：bootstrap区间、配对检验和真机审计摘要；
- `quantum_contribution_decision.json`：独立候选质量判定；
- `frozen_confirmatory_protocol.json`：后续确认性实验的冻结阈值与决策规则；
- `counts/`、`circuits/`、`mappings/`和`replay/`：逐任务原始证据；
- 一键重建脚本：从归档counts重新生成统计表和图形，不提交新真机任务。

软件测试共31项全部通过，Python依赖检查无冲突。该证据结构使评审者能够区分原始观测、确定性派生量和作者解释。

# 10 结论

本文提出面向CVRP局部精化的DeepBlock拓扑协同QAOA真机混合框架，并在Baihua上完成36个正式真机任务。通过预设最低10%BQM能量门槛，真机低能概率质量达到27.76%，为均匀随机的2.73倍，6/6个独立种子方向一致，探索性精确检验p=0.03125。这一结果证明Baihua量子采样对代理QUBO产生了明确的正向候选质量贡献。

在端到端层面，Baihua使6/6个正式实例的路线距离下降，平均恢复71.4%的局部精确余量；但平均绝对改进未超过随机和理想模拟，差异不显著。因此，本文不宣称量子加速，而将贡献定位为真实、可审计、可复现的NISQ混合局部精化，以及一套将量子候选质量与经典后处理效果分离的评测方法。

候选预算实验表明，当前真机样本中存在大量低频好解；代理相关性分析则指出，下一步的决定性任务是使QUBO能量与真实CVRP增量对齐。在冻结阈值的新批次确认、更多独立种子、跨日硬件重复和更大局部核完成之前，量子优势仍应保持开放而非预设。正是这种可证伪的边界，使本项目能够作为量子+优化赛道中兼顾工程完整性和科学可信度的竞赛方案。

# 参考文献

[1] Dantzig G B, Ramser J H. The Truck Dispatching Problem. Management Science, 1959, 6(1): 80-91. DOI: 10.1287/mnsc.6.1.80.

[2] Laporte G. Fifty Years of Vehicle Routing. Transportation Science, 2009, 43(4): 408-416. DOI: 10.1287/trsc.1090.0301.

[3] Preskill J. Quantum Computing in the NISQ Era and Beyond. Quantum, 2018, 2: 79. DOI: 10.22331/q-2018-08-06-79.

[4] Glover F, Kochenberger G, Du Y. Quantum Bridge Analytics I: A Tutorial on Formulating and Using QUBO Models. 4OR, 2019, 17: 335-371. DOI: 10.1007/s10288-019-00424-y.

[5] Lucas A. Ising Formulations of Many NP Problems. Frontiers in Physics, 2014, 2: 5. DOI: 10.3389/fphy.2014.00005.

[6] Farhi E, Goldstone J, Gutmann S. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028, 2014.

[7] Hadfield S, Wang Z, O'Gorman B, et al. From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. Algorithms, 2019, 12(2): 34. DOI: 10.3390/a12020034.

[8] Zhou L, Wang S T, Choi S, et al. Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices. Physical Review X, 2020, 10: 021067. DOI: 10.1103/PhysRevX.10.021067.

[9] Cerezo M, Arrasmith A, Babbush R, et al. Variational Quantum Algorithms. Nature Reviews Physics, 2021, 3: 625-644. DOI: 10.1038/s42254-021-00348-9.

[10] Xu H Z, Zhuang W F, Wang Z A, et al. Quafu-Qcover: Explore Combinatorial Optimization Problems on Cloud-Based Quantum Computers. Chinese Physics B, 2024, 33(5): 050302. DOI: 10.1088/1674-1056/ad18ab.

[11] Feld S, Roch C, Gabor T, et al. A Hybrid Solution Method for the Capacitated Vehicle Routing Problem Using a Quantum Annealer. Frontiers in ICT, 2019, 6: 13. DOI: 10.3389/fict.2019.00013.

[12] Harwood S, Gambella C, Trenev D, et al. Formulating and Solving Routing Problems on Quantum Computers. IEEE Transactions on Quantum Engineering, 2021, 2: 3100118. DOI: 10.1109/TQE.2021.3049230.

[13] Azad U, Behera B K, Ahmed E A, et al. Solving Vehicle Routing Problem Using Quantum Approximate Optimization Algorithm. IEEE Transactions on Intelligent Transportation Systems, 2023, 24: 7564-7573. DOI: 10.1109/TITS.2022.3172241.

[14] Fitzek D, Ghandriz T, Laine L, et al. Applying Quantum Approximate Optimization to the Heterogeneous Vehicle Routing Problem. Scientific Reports, 2024, 14: 25415. DOI: 10.1038/s41598-024-76967-w.

[15] Xie N, Lee X, Cai D, et al. A Feasibility-Preserved Quantum Approximate Solver for the Capacitated Vehicle Routing Problem. arXiv:2308.08785, 2023.

[16] Google. OR-Tools: Capacity Constraints for Vehicle Routing. https://developers.google.com/optimization/routing/cvrp, accessed 2026-08-01.

[17] Huang W H, Matsuyama H, Yamashiro Y. Solving Capacitated Vehicle Routing Problem with Quantum Alternating Operator Ansatz and Column Generation. arXiv:2503.17051, 2025.

# 附录A 关键任务与结果目录

正式Baihua p=1、k=64真机批次分别保存在：

- `results/baihua_hardware_p1_k64_seeds2_3_4_20260801`
- `results/baihua_hardware_p1_k64_seeds7_9_10_20260801`

候选质量与竞赛证据包保存在：

- `results/baihua_quantum_candidate_quality_20260801`
- `results/baihua_competition_package_20260801`

一键重建命令：

`powershell -ExecutionPolicy Bypass -File experiments/rebuild_baihua_competition_package.ps1`

# 附录B 后续确认性实验协议

后续确认性批次必须在提交新真机任务前冻结以下内容：

1. 高质量候选定义仍为每个QUBO最低10%能量秩，不改变为5%、20%或事后最优分位数。
2. 主统计单位为独立种子，不把子问题或shots当作独立样本。
3. 主比较为同一QUBO完整状态空间上的均匀随机分布。
4. 判定规则为硬件-随机低能概率质量差的95%种子bootstrap下界大于0，且双侧精确配对p<0.05。
5. 路线改善和组合门槛只作为下游诊断，不改变量子候选质量判定。
6. 所有预注册种子进入主结果；无局部余量实例按零改善纳入总体端到端统计。

