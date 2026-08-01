# Quantum Route Forge

面向容量约束车辆路径问题（CVRP）的量子—经典混合原型。当前主流程已经从原始 BQM 分配方案重构为：

```text
容量约束 K-Means
    → 自适应 2–6 客户稀疏 Max-Cut
    → Quafu/本地经典采样
    → Tabu 接受与容量修复
    → OR-Tools 单车 TSP
```

原 BQM + 模拟退火实现仍保留为 `classical` 基线，不再承担主量子流程。

## 核心设计

1. **容量约束聚类**  
   使用需求加权空间特征和多次重启，将全部客户分配给车辆；每个客户恰好出现一次，每辆车载荷不超过容量。

2. **自适应 Tabu-QAOA**  
   每轮选择一对车辆及其边界客户。`qaoa_subproblem_size=6` 表示“最多 6 个客户”，不是“必须 6 个”。如果当前车辆对只有 4 或 5 个客户，就直接构造 4 或 5 比特电路，不填充虚拟客户，也不丢弃该车辆对。

3. **硬件安全的稀疏 Max-Cut**  
   子图最多 6 个顶点、10 条距离加权边。`p=1` QAOA 使用显式 `CX-RZ-CX` 代价层，因此 CNOT 数为 `2 × 边数`，上限 20。

4. **Quafu 真机适配**  
   Python 3.12 环境优先使用 QuarkStudio 的 `Task(token).run(task)`；任务以 OpenQASM 2.0 提交，编译器为 `qsteed`，默认芯片为 `Dongling`，shots 自动归一化为 1024 的倍数。网络或结果获取失败时，会记录原因并回退到本地精确 Max-Cut。

5. **严格可行性和路线求解**  
   Tabu 候选只接受容量可行分配，随后再次执行容量修复。每辆车内部使用 OR-Tools 求近似 TSP；若 OR-Tools 不可用，可回退到最近邻 + 2-opt。

## 自适应分层消融

搜索完整性优先于固定电路宽度。本项目不为实验整齐而强制 6 比特，而是按每轮实际子问题规模 `sub_k` 分层。

每个量子迭代还会计算一个**经典精确 Max-Cut 影子解**。影子解与量子解共享同一轮次、车辆对、客户集合、稀疏图、当前分配和 Tabu 状态，但不改变主搜索轨迹。因此层内 `B − A` 是逐轮配对差值：

- B：配置的量子/硬件采样提案；
- A：同一子问题上的经典精确 Max-Cut 影子提案；
- `B − A > 0`：量子提案在路线代理改善量上更好；
- 统计按实际 `sub_k` 分组，通常重点报告 4、5、6 比特层。

每轮 JSON 记录包含：

```json
{
  "iteration": 5,
  "sub_k": 5,
  "n_edges": 7,
  "sub_cnot": 14,
  "vehicles": [3, 7],
  "cut_before": 120.0,
  "cut_proposed": 135.0,
  "cut_after": 135.0,
  "proxy_before": 42.1,
  "proxy_after": 40.8,
  "quantum_improvement": 1.3,
  "classical_improvement": 1.0,
  "paired_improvement_delta": 0.3
}
```

影子对照用于量子模块的逐轮配对消融，不等同于一条独立演化的经典完整路线。论文中还应同时报告 `hybrid_local` 和 `classical` 的端到端结果，避免混淆模块级与系统级结论。完整协议见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

## 环境安装

QuarkStudio 真机接口要求 Python 3.12。Windows PowerShell：

```powershell
cd "E:\222CVRP - GPT\Project Source Code"
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip check
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后只在 `.env` 中填写：

```dotenv
QUAFU_API_TOKEN=your_token_here
QUAFU_CHIP=Dongling
```

`.env` 已被 `.gitignore` 排除。CLI 和 Web UI 会自动读取它，日志和 JSON 不输出 token。

## CLI

先用本地精确 Max-Cut 验证完整混合流程：

```powershell
.\.venv312\Scripts\python.exe run_cli.py `
  --mode hybrid_local `
  --customers 18 --vehicles 4 --capacity 28 --seed 2026 `
  --tabu-iterations 20 `
  --qaoa-subproblem-size 6 `
  --routing-method ortools `
  --output-json results\adaptive_local.json
```

提交 Quafu 真机：

```powershell
.\.venv312\Scripts\python.exe run_cli.py `
  --mode quantum `
  --customers 18 --vehicles 4 --capacity 28 --seed 2026 `
  --tabu-iterations 20 `
  --qaoa-subproblem-size 6 `
  --qaoa-max-edges 10 `
  --quafu-backend Dongling `
  --quafu-shots 1024 `
  --output-json results\adaptive_hardware.json
```

生成按 `sub_k` 分层的 JSON/CSV 表：

```powershell
.\.venv312\Scripts\python.exe tools\summarize_ablation.py `
  results\adaptive_hardware.json `
  --output-json results\adaptive_hardware_layers.json `
  --output-csv results\adaptive_hardware_layers.csv
```

主要模式：

| 模式 | 分配与改进 | QAOA 来源 | 路线 |
|---|---|---|---|
| `quantum` | 容量聚类 + Tabu | Quafu 真机，失败时本地回退 | OR-Tools |
| `hybrid_local` | 容量聚类 + Tabu | 本地精确 Max-Cut | OR-Tools |
| `classical` | 原 BQM 基线 | 无 | OR-Tools/启发式 |

## Web UI

```powershell
.\.venv312\Scripts\python.exe app.py --port 8050
```

打开 `http://127.0.0.1:8050`。UI 默认采用自适应窗口，`QAOA Customers=6` 表示上限为 6。

## 验证

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe tools\check_quafu.py
```

只有在需要提交最小 Bell 电路时才加：

```powershell
.\.venv312\Scripts\python.exe tools\check_quafu.py --submit-bell
```

已完成的真机连通性和历史实验审计见 [docs/REAL_HARDWARE_VALIDATION.md](docs/REAL_HARDWARE_VALIDATION.md)。历史固定 6 比特运行仅保留为审计记录，不作为当前自适应算法的直接对照。

## 目录

```text
Project Source Code/
├── app.py
├── run_cli.py
├── requirements.txt
├── src/quantum_route_forge/
│   ├── clustering.py
│   ├── maxcut_qaoa.py
│   ├── tabu_qaoa.py
│   ├── ablation.py
│   ├── quafu_bridge.py
│   ├── routing.py
│   └── pipeline.py
├── tools/
│   ├── check_quafu.py
│   ├── archive_task_counts.py
│   └── summarize_ablation.py
├── tests/
└── docs/
```

## 结论边界

单个实例或单次真机运行不能证明“量子优势”。可报告的是：任务确实提交到真机、测量 counts 确实进入闭环、容量和门数约束满足，以及各 `sub_k` 层内的配对统计。量子优越性需要更多种子、置信区间和独立端到端基线支持。

## Baihua DeepBlock（8 比特深层 QAOA）

DeepBlock 是独立新增入口，不改变现有 Shenglian 10/16 比特流程。它把约 16 个边界客户构造成三个最多 8 客户、默认重叠 3 客户的局部块，执行 `B1→B2→B3→B3→B2→B1` 扫描。量子代理仅生成候选；容量修复、路线距离和接受决定仍由经典完整目标负责。

本地模拟与硬件 dry-run（默认不提交真机）：

```powershell
.\.venv312\Scripts\python.exe experiments\run_baihua_deepblock.py `
  --seeds 3 --pool-size 16 --block-size 8 --overlap 3 `
  --qaoa-depth 2 --shots 4096 --candidate-k 8 `
  --arms random,sim,baihua `
  --outdir results\baihua_deepblock_p2
```

读取当前 Baihua 标定并生成物理映射/编译审计：

```powershell
.\.venv312\Scripts\python.exe experiments\run_baihua_deepblock.py `
  --seeds 3 --arms baihua --backend Baihua --fetch-calibration `
  --outdir results\baihua_deepblock_dryrun
```

只有同时传入两个显式开关时才允许提交真机：

```powershell
.\.venv312\Scripts\python.exe experiments\run_baihua_deepblock.py `
  --seeds 1 --arms baihua --backend Baihua --fetch-calibration `
  --submit-hardware --confirm-hardware-submit `
  --outdir results\baihua_deepblock_hardware_smoke
```

深度扫描、离线回放和分层统计：

```powershell
.\.venv312\Scripts\python.exe experiments\run_baihua_depth_scan.py `
  --seeds 3 --depths 1,2,3 --shots 4096 `
  --outdir results\baihua_depth_scan

.\.venv312\Scripts\python.exe experiments\replay_baihua_counts.py `
  --input results\baihua_deepblock_p2 --verify

.\.venv312\Scripts\python.exe experiments\analyze_baihua_deepblock.py `
  --input results\baihua_deepblock_p2 --split-by-headroom
```

输出目录包含 `config.json`、`calibration_snapshot.json`、实例与子问题 CSV、原始 counts、逻辑/物理线路、映射、逐子问题 replay 记录、`analysis.json` 和 `report.md`。`random`、`sim`、`baihua` 共用 shots、`candidate_k`、过滤、修复和接受口径；`exact` 只用于判断局部精化空间。

## Baihua competition evidence package

The formal p=1, k=64 package uses six pre-screened seeds with local-refinement headroom, 36 Baihua tasks, 4096 shots per task, paired random/simulator/exact arms, and replay verification. Rebuild every derived CSV, statistic, and figure from the archived raw counts without submitting new hardware jobs:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\rebuild_baihua_competition_package.ps1
```

The main report is `results/baihua_competition_package_20260801/competition_report.md`. The result supports useful positive hardware-assisted refinement, but does not claim quantum speedup or superiority over the random baseline.
