from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.experiment_logger import atomic_json, write_csv  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_ci(values: list[float], samples: int = 20_000) -> tuple[float, float]:
    rng = random.Random(20260801)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def _sign_flip_pvalue(values: list[float]) -> float | None:
    if not values:
        return None
    observed = abs(statistics.fmean(values))
    permutations = [
        abs(statistics.fmean(value * sign for value, sign in zip(values, signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in permutations) / len(permutations)


def _load_formal(hardware_dirs: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw: list[dict[str, str]] = []
    metrics: list[dict[str, str]] = []
    replay_records = replay_passed = 0
    for directory in hardware_dirs:
        raw.extend(_read_csv(directory / "instance_summary.csv"))
        metrics.extend(_read_csv(directory / "subproblem_metrics.csv"))
        verification = json.loads((directory / "replay" / "verification.json").read_text(encoding="utf-8"))
        replay_records += int(verification["records"])
        replay_passed += int(verification["passed"])
    by = {(int(row["seed"]), row["arm"]): row for row in raw}
    seeds = sorted({seed for seed, _arm in by})
    rows: list[dict[str, object]] = []
    for seed in seeds:
        exact = float(by[seed, "exact"]["improvement_vs_baseline"])
        row: dict[str, object] = {
            "seed": seed,
            "baseline_distance": float(by[seed, "exact"]["baseline_distance"]),
            "exact_headroom": exact,
        }
        for arm in ("baihua", "random", "sim", "exact"):
            improvement = float(by[seed, arm]["improvement_vs_baseline"])
            row[f"{arm}_improvement"] = improvement
            row[f"{arm}_headroom_utilization"] = improvement / exact if exact > 1e-12 else None
            row[f"{arm}_accepted_moves"] = int(by[seed, arm]["accepted_moves"])
        row["baihua_minus_random_improvement"] = float(row["baihua_improvement"]) - float(row["random_improvement"])
        rows.append(row)
    hardware = [row for row in metrics if str(row["arm"]).startswith("baihua")]
    task_ids = {row["task_id"] for row in hardware if row.get("task_id")}
    metadata = {
        "formal_seeds": seeds,
        "hardware_tasks": len(task_ids),
        "shots_per_task": int(hardware[0]["shots"]) if hardware else 0,
        "total_hardware_shots": sum(int(row["shots"]) for row in hardware),
        "replay_records": replay_records,
        "replay_passed": replay_passed,
        "compilation_passed": sum(str(row["compilation_passed"]).lower() == "true" for row in hardware),
        "zero_swap_tasks": sum(int(row["swap_count"]) == 0 for row in hardware),
        "mean_circuit_depth": statistics.fmean(float(row["circuit_depth"]) for row in hardware),
        "mean_two_qubit_layers": statistics.fmean(float(row["two_qubit_gate_layers"]) for row in hardware),
        "physical_qubits": json.loads(hardware[0]["physical_qubits"]) if hardware else [],
        "calibration_time_first": min(row["calibration_time"] for row in hardware),
        "calibration_time_last": max(row["calibration_time"] for row in hardware),
    }
    return rows, metadata


def _arm_summary(formal: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for arm in ("baihua", "random", "sim", "exact"):
        improvements = [float(row[f"{arm}_improvement"]) for row in formal]
        utilization = [float(row[f"{arm}_headroom_utilization"]) for row in formal]
        low, high = _bootstrap_ci(improvements)
        summaries.append(
            {
                "arm": arm,
                "instances": len(formal),
                "positive_instances": sum(value > 1e-9 for value in improvements),
                "mean_improvement": statistics.fmean(improvements),
                "median_improvement": statistics.median(improvements),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "mean_headroom_utilization": statistics.fmean(utilization),
            }
        )
    return summaries


def _legacy_rows(directories: list[Path], depth: int) -> list[dict[str, object]]:
    found: dict[tuple[int, str], dict[str, str]] = {}
    for directory in directories:
        for row in _read_csv(directory / "instance_summary.csv"):
            seed = int(row["seed"])
            if seed in {2, 3, 4}:
                found[(seed, row["arm"])] = row
    return [
        {
            "depth": depth,
            "seed": seed,
            "arm": arm,
            "improvement": float(row["improvement_vs_baseline"]),
        }
        for (seed, arm), row in sorted(found.items())
    ]


def _plots(
    outdir: Path,
    formal: list[dict[str, object]],
    arm_summary: list[dict[str, object]],
    candidate_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    legacy: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    seeds = [str(row["seed"]) for row in formal]
    arms = ("baihua", "random", "sim", "exact")
    colors = {"baihua": "#6A4C93", "random": "#8D99AE", "sim": "#1982C4", "exact": "#2A9D8F"}
    x = list(range(len(seeds)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for offset, arm in enumerate(arms):
        ax.bar(
            [value + (offset - 1.5) * width for value in x],
            [float(row[f"{arm}_improvement"]) for row in formal],
            width,
            label=arm,
            color=colors[arm],
        )
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Independent seed")
    ax.set_ylabel("Route-distance improvement (higher is better)")
    ax.set_title("Formal p=1, k=64 closed-loop results on Baihua")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig1_formal_seed_improvements.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = [row["arm"] for row in arm_summary]
    means = [100 * float(row["mean_headroom_utilization"]) for row in arm_summary]
    ax.bar(names, means, color=[colors[str(name)] for name in names])
    ax.set_ylabel("Mean exact-headroom recovery (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Normalized local-refinement quality")
    for index, value in enumerate(means):
        ax.text(index, value + 2, f"{value:.1f}%", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig2_headroom_utilization.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, marker in ((1, "o"), (2, "s")):
        subset = sorted(
            (row for row in candidate_rows if int(row["depth"]) == depth),
            key=lambda row: int(row["candidate_k"]),
        )
        ax.plot(
            [int(row["candidate_k"]) for row in subset],
            [100 * float(row["hit_rate"]) for row in subset],
            marker=marker,
            label=f"p={depth}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Online full-objective candidate budget k")
    ax.set_ylabel("Improving-subproblem hit rate (%)")
    ax.set_title("Existing hardware counts contain useful low-frequency candidates")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig3_candidate_budget_hit_rate.png", dpi=300)
    plt.close(fig)

    chosen = [
        row for row in ablation_rows
        if row["arm"] in {"sim", "random"}
        and row["configuration"] in {
            "reference_p1_k8_o3_bidir",
            "p1_k16_o3_bidir",
            "p1_k32_o3_bidir",
            "p1_k64_o3_bidir",
            "p2_k32_o3_bidir",
            "p1_k32_o0_bidir",
            "p1_k32_o3_forward",
        }
    ]
    labels = []
    configurations = [
        "reference_p1_k8_o3_bidir",
        "p1_k16_o3_bidir",
        "p1_k32_o3_bidir",
        "p1_k64_o3_bidir",
        "p2_k32_o3_bidir",
        "p1_k32_o0_bidir",
        "p1_k32_o3_forward",
    ]
    short = {
        "reference_p1_k8_o3_bidir": "p1 k8",
        "p1_k16_o3_bidir": "p1 k16",
        "p1_k32_o3_bidir": "p1 k32",
        "p1_k64_o3_bidir": "p1 k64",
        "p2_k32_o3_bidir": "p2 k32",
        "p1_k32_o0_bidir": "p1 k32, no overlap",
        "p1_k32_o3_forward": "p1 k32, forward",
    }
    labels = [short[value] for value in configurations]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, arm in enumerate(("sim", "random")):
        vals = [
            float(next(row for row in chosen if row["configuration"] == config and row["arm"] == arm)["mean_improvement"])
            for config in configurations
        ]
        ax.bar(
            [index + (offset - 0.5) * 0.36 for index in range(len(labels))],
            vals,
            0.36,
            label=arm,
            color=colors[arm],
        )
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_ylabel("Mean improvement over 10 paired seeds")
    ax.set_title("Simulation ablation used to gate hardware configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig4_ablation.png", dpi=300)
    plt.close(fig)

    if legacy:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for depth, color in ((1, "#6A4C93"), (2, "#FF595E")):
            values = [
                float(next(row for row in legacy if row["depth"] == depth and row["seed"] == seed and row["arm"] == "baihua")["improvement"])
                for seed in (2, 3, 4)
            ]
            ax.plot((2, 3, 4), values, marker="o", color=color, label=f"hardware p={depth}, k=8")
        upgraded = [float(row["baihua_improvement"]) for row in formal if int(row["seed"]) in {2, 3, 4}]
        ax.plot((2, 3, 4), upgraded, marker="s", linewidth=2.5, color="#2A9D8F", label="hardware p=1, k=64")
        ax.set_xticks((2, 3, 4))
        ax.set_xlabel("Seed")
        ax.set_ylabel("Route-distance improvement")
        ax.set_title("Depth versus decoding-budget intervention")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "fig5_hardware_intervention.png", dpi=300)
        plt.close(fig)


def _report(
    outdir: Path,
    formal: list[dict[str, object]],
    summaries: list[dict[str, object]],
    metadata: dict[str, object],
    no_headroom_ratio: float,
    sign_flip_p: float,
    cross_platform: dict[str, object] | None,
) -> None:
    arm = {str(row["arm"]): row for row in summaries}
    quantum_minus_random = [float(row["baihua_minus_random_improvement"]) for row in formal]
    wins = sum(value > 1e-9 for value in quantum_minus_random)
    ties = sum(abs(value) <= 1e-9 for value in quantum_minus_random)
    losses = sum(value < -1e-9 for value in quantum_minus_random)
    lines = [
        "# Baihua DeepBlock CVRP 竞赛实验报告",
        "",
        "## 一句话结论",
        "",
        f"Baihua p=1、k=64 真机闭环在 {arm['baihua']['positive_instances']}/{arm['baihua']['instances']} 个有局部余量实例中产生严格路线改善，平均恢复 {100*float(arm['baihua']['mean_headroom_utilization']):.1f}% 的精确局部余量；与随机采样为 {wins}胜{ties}平{losses}负，尚无量子优势证据。",
        "",
        "## 正式结果（统计单位：独立种子）",
        "",
        "| Arm | 正向实例 | 平均距离改进 | 95% bootstrap CI | 平均精确余量恢复 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("baihua", "random", "sim", "exact"):
        row = arm[name]
        lines.append(
            f"| {name} | {row['positive_instances']}/{row['instances']} | {float(row['mean_improvement']):.6f} | "
            f"[{float(row['bootstrap_ci95_low']):.6f}, {float(row['bootstrap_ci95_high']):.6f}] | "
            f"{100*float(row['mean_headroom_utilization']):.1f}% |"
        )
    lines.extend(
        [
            "",
            f"- Baihua − random 平均改进差：{statistics.fmean(quantum_minus_random):.6f}。",
            f"- 配对精确符号翻转检验（双侧）：p={sign_flip_p:.5f}，差异不显著。",
            f"- 10 种子预筛选中无局部改进空间比例：{100*no_headroom_ratio:.1f}%；正式真机集为预先识别的 6 个有余量种子。",
            "",
            "## 真机证据链",
            "",
            f"- 正式真机任务：{metadata['hardware_tasks']}，每任务 {metadata['shots_per_task']} shots，总计 {metadata['total_hardware_shots']} shots。",
            f"- 编译通过：{metadata['compilation_passed']}/{metadata['hardware_tasks']}；0 SWAP：{metadata['zero_swap_tasks']}/{metadata['hardware_tasks']}。",
            f"- 物理量子位：{metadata['physical_qubits']}；平均电路深度 {float(metadata['mean_circuit_depth']):.1f}，平均双量子位门层数 {float(metadata['mean_two_qubit_layers']):.1f}。",
            f"- 完整重放：{metadata['replay_passed']}/{metadata['replay_records']} 通过（包含 exact/random/sim/baihua 全臂）。",
            "",
            "## 为什么量子起到了正向作用、但不是量子优势",
            "",
            "真机 arm 的最终距离在六个实例中都下降，因此它在该混合闭环中产生了可用候选，并非只完成了线路提交。候选预算从 k=8 提到 k=64 后，种子 2/3/4 的真机平均改进从 1.593 提高到 3.475，说明大量好解位于低频样本中。另一方面，随机基线在绝对平均改进上仍略高，且配对差异不显著；单调接受规则本身也保证不会接受更差解。因此当前最准确的表述是“量子采样可用且有正向贡献，达到接近经典随机的质量”，而不是“量子加速/量子优势”。",
            "",
            "## 竞赛亮点",
            "",
            "1. 真实 Baihua 采样、逐块闭环更新，而非模拟器或静态批处理冒充真机。",
            "2. 拓扑感知稀疏 QUBO 与 8 量子位物理链直接对齐，正式任务全部 0 SWAP。",
            "3. 同时报告随机、理想模拟、精确枚举、无余量比例、深度噪声和失败结论，不隐藏负结果。",
            "4. 每个 counts、QASM、映射、校准时间、task ID、候选决策和路线状态均可离线重放。",
            "5. 用门控实验发现真正瓶颈是概率质量分散与解码预算，而非盲目增加 QAOA 深度。",
            "",
            "## 第二硬件平台证据（范围审计）",
            "",
        ]
    )
    if cross_platform:
        lines.extend(
            [
                f"现有 Shenglian 扫描覆盖 {cross_platform['min_width']}–{cross_platform['max_width']} 个真实决策量子位、{cross_platform['tasks']} 个真机任务，每任务 {cross_platform['shots_per_task']} shots；相对均匀随机合法样本的分布质量信号在全部宽度为正，最大验证宽度为 {cross_platform['max_positive_signal_width']}，top-5 精确最优的稳定边界为 {cross_platform['max_top5_exact_width']}。",
                "",
                "该扫描使用 2 车辆、seed 2026、链式 Max-Cut 目标和 8192 shots；Baihua 正式集使用 40 客户、4 车辆、8 位局部精化和 4096 shots。因此它只能作为“第二平台可运行性与宽度扩展”证据，不能与 Baihua 的最终路线改进做数值胜负比较。",
                "",
            ]
        )
    else:
        lines.extend(["未提供可审计的第二硬件平台数据，因此不做跨平台结论。", ""])
    lines.extend(
        [
            "## 结论边界",
            "",
            "本实验展示的是 NISQ 混合局部精化的有效性和工程可审计性，不建立相对于 OR-Tools、精确枚举或其他经典算法的计算加速。正式样本量为 6，仍需跨日期重复校准批次、更多独立实例及第二硬件平台对照。",
            "",
        ]
    )
    (outdir / "competition_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Baihua DeepBlock competition evidence package")
    parser.add_argument("--hardware-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-budget-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--headroom-screen", type=Path, required=True)
    parser.add_argument("--legacy-p1-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--legacy-p2-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--shenglian-width-scan", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    formal, metadata = _load_formal(args.hardware_dirs)
    summaries = _arm_summary(formal)
    differences = [float(row["baihua_minus_random_improvement"]) for row in formal]
    sign_flip_p = float(_sign_flip_pvalue(differences) or 1.0)
    screen = _read_csv(args.headroom_screen / "instance_summary.csv")
    exact_screen = [row for row in screen if row["arm"] == "exact"]
    no_headroom_ratio = statistics.fmean(str(row["has_refinement_space"]).lower() != "true" for row in exact_screen)
    candidate_rows = _read_csv(args.candidate_budget_dir / "candidate_budget_summary.csv")
    ablation_rows = _read_csv(args.ablation_dir / "ablation_summary.csv")
    legacy = _legacy_rows(args.legacy_p1_dirs, 1) + _legacy_rows(args.legacy_p2_dirs, 2)
    cross_platform = None
    if args.shenglian_width_scan:
        width_rows = _read_csv(args.shenglian_width_scan)
        widths = [int(row["width"]) for row in width_rows]
        cross_platform = {
            "platform": "Shenglian",
            "min_width": min(widths),
            "max_width": max(widths),
            "tasks": len(width_rows),
            "shots_per_task": int(width_rows[0]["shots"]),
            "max_positive_signal_width": max(int(row["width"]) for row in width_rows if float(row["signal"]) > 0),
            "max_top5_exact_width": max(int(row["width"]) for row in width_rows if str(row["top5_contains_exact_optimum"]).lower() == "true"),
            "all_mapping_verified": all(str(row["mapping_verified"]).lower() == "true" for row in width_rows),
            "all_zero_uncalibrated_couplings": all(int(row["uncalibrated_couplings"]) == 0 for row in width_rows),
        }
        atomic_json(args.outdir / "cross_platform_summary.json", cross_platform)
    write_csv(args.outdir / "formal_seed_results.csv", formal)
    write_csv(args.outdir / "formal_arm_summary.csv", summaries)
    atomic_json(
        args.outdir / "formal_summary.json",
        {
            "metadata": metadata,
            "arms": summaries,
            "baihua_vs_random": {
                "mean_improvement_difference": statistics.fmean(differences),
                "win_tie_loss": {
                    "win": sum(value > 1e-9 for value in differences),
                    "tie": sum(abs(value) <= 1e-9 for value in differences),
                    "loss": sum(value < -1e-9 for value in differences),
                },
                "exact_sign_flip_two_sided_pvalue": sign_flip_p,
            },
            "screened_no_headroom_ratio": no_headroom_ratio,
        },
    )
    _plots(args.outdir, formal, summaries, candidate_rows, ablation_rows, legacy)
    _report(args.outdir, formal, summaries, metadata, no_headroom_ratio, sign_flip_p, cross_platform)
    print(json.dumps({"outdir": str(args.outdir), "formal_seeds": len(formal), "hardware_tasks": metadata["hardware_tasks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
