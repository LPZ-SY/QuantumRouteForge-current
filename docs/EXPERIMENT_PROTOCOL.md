# 自适应 Tabu-QAOA 分层消融协议

## 1. 研究问题

在不裁剪车辆对搜索空间的前提下，比较量子采样提案与经典精确 Max-Cut 提案在局部 CVRP 重分配中的效果。

主流程的 `qaoa_subproblem_size=6` 是上限。每轮实际量子比特数为：

```text
sub_k = min(6, 当前车辆对可用于交换的客户数)
```

只要车辆对合计客户数 `>= 2` 就保留；合计少于 2 时无法构造 Max-Cut，也不存在有效的两侧交换，因此跳到下一个可用车辆对而不消耗迭代。其余情况不填充虚拟客户，不因为不足 6 个客户而过滤。

## 2. 逐轮配对

每轮先固定以下上下文：

- 迭代编号；
- 车辆对及其顺序；
- 所选客户及顺序；
- 稀疏 Max-Cut 边和权重；
- 当前车辆分配；
- 当前 Tabu 表；
- 容量约束。

实验组 B 使用配置的采样器（真机时为 Quafu counts 的最高频 bitstring）。对照组 A 在完全相同的上下文上使用经典穷举精确 Max-Cut。两者都评估正向和反向分区，并采用相同的容量、Tabu 和 aspiration 规则。

经典影子解只用于当轮反事实比较，不更新 B 的搜索轨迹。这样可以保证严格配对，不会因两条轨迹提前分叉而破坏“相同客户、相同图”的条件。

## 3. 指标定义

每轮至少记录：

| 字段 | 定义 |
|---|---|
| `sub_k` | 实际客户数/量子比特数 |
| `n_edges` | 稀疏 Max-Cut 边数 |
| `sub_cnot` | QAOA 代价层 CNOT 数，等于 `2 × n_edges` |
| `vehicles` | 当前车辆对 |
| `cut_before` | 当前车辆归属对应的加权 cut |
| `cut_proposed` | B 的采样分区 cut |
| `cut_after` | 接受/拒绝后的实际 cut |
| `proxy_before` | 当轮前路线代理距离 |
| `proxy_after` | B 接受/拒绝后的路线代理距离 |
| `quantum_improvement` | `proxy_before - B候选距离` |
| `classical_improvement` | `proxy_before - A候选距离` |
| `paired_improvement_delta` | `quantum_improvement - classical_improvement` |

距离越小越好，因此 improvement 越大越好，`paired_improvement_delta > 0` 表示 B 在该轮优于 A。

## 4. 分层统计

按 `sub_k` 独立计算：

- 迭代数 `N_k`；
- 真机迭代数；
- 平均边数和 CNOT 数；
- B 平均改善；
- A 平均改善；
- 平均配对差 `mean(B − A)`；
- B/A 平均 cut。

4、5、6 比特是主要报告层，但若实际出现 2 或 3 比特，也必须保留并单独报告，不能静默丢弃。

推荐在多个随机种子上同时报告均值、标准差、配对置信区间和效应量。若某层样本太少，应明确标注，不把该层外推为总体结论。

## 5. 系统级对照

逐轮影子对照回答的是“在相同局部问题上，量子提案与经典提案谁更好”。它不回答完整算法端到端谁更好。

系统级实验应另行运行：

1. `quantum`：真机采样驱动的完整 Tabu 轨迹；
2. `hybrid_local`：经典精确 Max-Cut 驱动的完整 Tabu 轨迹；
3. `classical`：原 BQM 基线；
4. 容量聚类 + OR-Tools、Tabu 迭代为 0 的基线。

这些轨迹可能分叉，因此只比较最终距离、可行率、运行时间和资源成本，不声称它们逐轮使用相同客户子问题。

## 6. 可复现实验命令

```powershell
.\.venv312\Scripts\python.exe run_cli.py `
  --mode quantum `
  --seed 2026 --customers 18 --vehicles 4 --capacity 28 `
  --tabu-iterations 20 --qaoa-subproblem-size 6 `
  --qaoa-max-edges 10 --quafu-shots 1024 `
  --output-json results\adaptive_hardware.json

.\.venv312\Scripts\python.exe tools\summarize_ablation.py `
  results\adaptive_hardware.json `
  --output-json results\adaptive_hardware_layers.json `
  --output-csv results\adaptive_hardware_layers.csv
```

每个真机任务都应保留 task ID、backend、counts、QASM、实际 `sub_k`、边数和 CNOT 数。严禁在实验产物中保存 token。
