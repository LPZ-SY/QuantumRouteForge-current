from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _to_float(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(v))
    except Exception:
        return None


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制比赛论文图表")
    parser.add_argument("--figdir", type=str, default="paper/figures")
    parser.add_argument("--results", type=str, default="results/competition_runs/results.csv")
    args = parser.parse_args()

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results)
    energy_path = ROOT / "results/energy_distribution/energy_distribution.csv"
    topk_path = ROOT / "results/topk_refinement/topk_refinement.csv"
    exact_path = ROOT / "results/exact_runs/exact_comparison.csv"
    ablation_path = ROOT / "results/pipeline_ablation/pipeline_ablation.csv"
    quafu_diag_path = ROOT / "results/quafu_diagnostics/diagnostic_summary.json"

    rows = _read_csv(results_path)
    energy_rows = _read_csv(energy_path)
    topk_rows = _read_csv(topk_path)
    exact_rows = _read_csv(exact_path)
    ablation_rows = _read_csv(ablation_path)

    quafu_diag = {}
    if quafu_diag_path.exists():
        try:
            quafu_diag = json.loads(quafu_diag_path.read_text(encoding="utf-8"))
        except Exception:
            quafu_diag = {}

    manifest: dict[str, Any] = {}

    # 1) 系统架构图
    try:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        ax.axis("off")
        labels = [
            "场景输入",
            "BQM 建模",
            "求解器接口",
            "Quafu 提交 / 经典求解",
            "分配解码与修复",
            "路线细化",
            "可视化输出",
        ]
        xs = [0.03, 0.17, 0.31, 0.48, 0.66, 0.81, 0.93]
        for x, label in zip(xs, labels):
            ax.text(
                x,
                0.5,
                label,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eaf3ff", "edgecolor": "#4a7ebb"},
                fontsize=11,
                transform=ax.transAxes,
            )
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.04, 0.5),
                xytext=(xs[i] + 0.05, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#355f8a"},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
        ax.set_title("Quantum Route Forge 系统流程", fontsize=14)
        _save(fig, figdir / "fig_system_architecture.png")
        manifest["fig_system_architecture.png"] = {
            "success": True,
            "source": "流程示意图（固定绘制）",
            "fields": ["场景输入", "BQM 建模", "Quafu/经典", "解码修复", "路线细化"],
        }
    except Exception as exc:
        manifest["fig_system_architecture.png"] = {"success": False, "reason": str(exc)}

    def _mode_name(x: str) -> str:
        if x == "classical":
            return "classical"
        if x == "random":
            return "random"
        if x == "quantum":
            return "quantum"
        return x

    valid_rows = [r for r in rows if _to_float(r.get("total_route_distance")) is not None]

    # 2) 客户数量-总路线距离
    try:
        if valid_rows:
            grp = defaultdict(list)
            for r in valid_rows:
                c = _to_int(r.get("customers"))
                m = _mode_name(str(r.get("mode", "")))
                d = _to_float(r.get("total_route_distance"))
                if c is None or d is None:
                    continue
                grp[(m, c)].append(d)

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            modes = sorted({k[0] for k in grp.keys()})
            for m in modes:
                xs = sorted({k[1] for k in grp.keys() if k[0] == m})
                ys = [_mean(grp[(m, x)]) for x in xs]
                ax.plot(xs, ys, marker="o", label=m)
            ax.set_title("不同客户规模下的总路线距离")
            ax.set_xlabel("客户数量")
            ax.set_ylabel("总路线距离")
            ax.legend(title="求解模式")
            ax.grid(alpha=0.25)
            _save(fig, figdir / "fig_total_distance_by_customers.png")
            manifest["fig_total_distance_by_customers.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["customers", "mode", "total_route_distance"],
            }
        else:
            manifest["fig_total_distance_by_customers.png"] = {"success": False, "reason": "缺少 results.csv 或距离字段"}
    except Exception as exc:
        manifest["fig_total_distance_by_customers.png"] = {"success": False, "reason": str(exc)}

    # 3) 客户数量-BQM 能量
    try:
        e_rows = [r for r in rows if _to_float(r.get("bqm_energy")) is not None]
        if e_rows:
            grp = defaultdict(list)
            for r in e_rows:
                c = _to_int(r.get("customers"))
                m = _mode_name(str(r.get("mode", "")))
                e = _to_float(r.get("bqm_energy"))
                if c is None or e is None:
                    continue
                grp[(m, c)].append(e)
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for m in sorted({k[0] for k in grp.keys()}):
                xs = sorted({k[1] for k in grp.keys() if k[0] == m})
                ys = [_mean(grp[(m, x)]) for x in xs]
                ax.plot(xs, ys, marker="o", label=m)
            ax.set_title("不同客户规模下的 BQM 能量")
            ax.set_xlabel("客户数量")
            ax.set_ylabel("BQM 能量")
            ax.legend(title="求解模式")
            ax.grid(alpha=0.25)
            _save(fig, figdir / "fig_energy_by_customers.png")
            manifest["fig_energy_by_customers.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["customers", "mode", "bqm_energy"],
            }
        else:
            manifest["fig_energy_by_customers.png"] = {"success": False, "reason": "缺少 bqm_energy"}
    except Exception as exc:
        manifest["fig_energy_by_customers.png"] = {"success": False, "reason": str(exc)}

    # 4) 客户数量-运行时间
    try:
        t_rows = [r for r in rows if _to_float(r.get("runtime_total_sec")) is not None]
        if t_rows:
            grp = defaultdict(list)
            for r in t_rows:
                c = _to_int(r.get("customers"))
                m = _mode_name(str(r.get("mode", "")))
                t = _to_float(r.get("runtime_total_sec"))
                if c is None or t is None:
                    continue
                grp[(m, c)].append(t)
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for m in sorted({k[0] for k in grp.keys()}):
                xs = sorted({k[1] for k in grp.keys() if k[0] == m})
                ys = [_mean(grp[(m, x)]) for x in xs]
                ax.plot(xs, ys, marker="o", label=m)
            ax.set_title("不同客户规模下的运行时间")
            ax.set_xlabel("客户数量")
            ax.set_ylabel("运行时间（秒）")
            ax.legend(title="求解模式")
            ax.grid(alpha=0.25)
            _save(fig, figdir / "fig_runtime_by_customers.png")
            manifest["fig_runtime_by_customers.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["customers", "mode", "runtime_total_sec"],
            }
        else:
            manifest["fig_runtime_by_customers.png"] = {"success": False, "reason": "缺少 runtime_total_sec"}
    except Exception as exc:
        manifest["fig_runtime_by_customers.png"] = {"success": False, "reason": str(exc)}

    # 5) 可行解比例
    try:
        if rows:
            grp = defaultdict(list)
            for r in rows:
                m = _mode_name(str(r.get("mode", "")))
                grp[m].append(1.0 if _to_bool(r.get("feasible")) else 0.0)
            modes = sorted(grp.keys())
            vals = [_mean(grp[m]) for m in modes]
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.bar(modes, vals, color=["#4e79a7", "#f28e2b", "#59a14f"][: len(modes)])
            ax.set_ylim(0, 1.0)
            ax.set_title("不同求解模式的可行解比例")
            ax.set_xlabel("求解模式")
            ax.set_ylabel("可行解比例")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_feasible_rate.png")
            manifest["fig_feasible_rate.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["mode", "feasible"],
            }
        else:
            manifest["fig_feasible_rate.png"] = {"success": False, "reason": "缺少 results.csv"}
    except Exception as exc:
        manifest["fig_feasible_rate.png"] = {"success": False, "reason": str(exc)}

    # 6) BQM 规模扩展图
    try:
        if rows:
            grp_vars = defaultdict(list)
            grp_quads = defaultdict(list)
            for r in rows:
                c = _to_int(r.get("customers"))
                nv = _to_float(r.get("num_bqm_variables"))
                nq = _to_float(r.get("num_bqm_quadratic_terms"))
                if c is None or nv is None or nq is None:
                    continue
                grp_vars[c].append(nv)
                grp_quads[c].append(nq)
            xs = sorted(grp_vars.keys())
            if xs:
                yv = [_mean(grp_vars[x]) for x in xs]
                yq = [_mean(grp_quads[x]) for x in xs]
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                ax.plot(xs, yv, marker="o", label="BQM 变量数")
                ax.plot(xs, yq, marker="s", label="BQM 二次项数量")
                ax.set_title("BQM 规模随客户数量变化")
                ax.set_xlabel("客户数量")
                ax.set_ylabel("规模")
                ax.legend()
                ax.grid(alpha=0.25)
                _save(fig, figdir / "fig_bqm_size_scaling.png")
                manifest["fig_bqm_size_scaling.png"] = {
                    "success": True,
                    "source": str(results_path),
                    "fields": ["customers", "num_bqm_variables", "num_bqm_quadratic_terms"],
                }
            else:
                manifest["fig_bqm_size_scaling.png"] = {"success": False, "reason": "无 BQM 规模字段"}
        else:
            manifest["fig_bqm_size_scaling.png"] = {"success": False, "reason": "缺少 results.csv"}
    except Exception as exc:
        manifest["fig_bqm_size_scaling.png"] = {"success": False, "reason": str(exc)}

    # 7) Quafu 提交状态统计
    try:
        qrows = [r for r in rows if str(r.get("mode", "")) == "quantum"]
        if qrows:
            status_count = defaultdict(int)
            for r in qrows:
                st = str(r.get("quafu_status", "unknown")).strip() or "unknown"
                if st == "submitted":
                    label = "Quafu 提交成功"
                elif st == "fallback":
                    label = "经典回退"
                elif st == "failed":
                    label = "提交失败"
                elif st == "token_missing":
                    label = "缺少 token"
                elif st == "timeout":
                    label = "超时"
                else:
                    label = st
                status_count[label] += 1
            labels = list(status_count.keys())
            vals = [status_count[k] for k in labels]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, vals)
            ax.set_title("Quantum 模式 Quafu 提交状态统计")
            ax.set_xlabel("状态")
            ax.set_ylabel("次数")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_quafu_submission_status.png")
            manifest["fig_quafu_submission_status.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["mode", "quafu_status"],
            }
        else:
            manifest["fig_quafu_submission_status.png"] = {"success": False, "reason": "无 quantum 模式数据"}
    except Exception as exc:
        manifest["fig_quafu_submission_status.png"] = {"success": False, "reason": str(exc)}

    # 8) 代表性路线图
    try:
        case_candidates = [
            r
            for r in rows
            if _to_int(r.get("seed")) == 2026
            and _to_int(r.get("customers")) == 48
            and _to_int(r.get("vehicles")) == 4
            and str(r.get("routes", "")).strip()
        ]
        if case_candidates:
            case = sorted(case_candidates, key=lambda r: _to_float(r.get("total_route_distance")) or 1e18)[0]
            coords = json.loads(case.get("coordinates", "{}"))
            routes = json.loads(case.get("routes", "[]"))

            fig, ax = plt.subplots(figsize=(7, 7))
            depot = coords.get("depot", [0, 0])
            customers = coords.get("customers", [])
            ax.scatter([depot[0]], [depot[1]], marker="*", s=220, c="black", label="仓库")
            ax.scatter([c["x"] for c in customers], [c["y"] for c in customers], s=18, c="#5f83a8", alpha=0.8, label="客户")

            id2xy = {int(c["customer_id"]): (float(c["x"]), float(c["y"])) for c in customers}
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#9467bd"]
            for i, rt in enumerate(routes):
                ids = [int(x) for x in rt.get("customer_ids", [])]
                pts = [depot] + [id2xy[x] for x in ids if x in id2xy] + [depot]
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2, label=f"车辆{rt.get('vehicle_id')}")

            title = (
                f"代表性路线图（seed=2026, customers=48, vehicles=4）\n"
                f"mode={case.get('mode')} used={case.get('used_solver')} distance={case.get('total_route_distance')} capacity={case.get('capacity')}"
            )
            if str(case.get("mode")) == "quantum":
                title += (
                    f"\ntask_id={case.get('quafu_task_id')} backend={case.get('quafu_backend')} status={case.get('quafu_status')}"
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X 坐标")
            ax.set_ylabel("Y 坐标")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.2)
            _save(fig, figdir / "case_route_visualization.png")
            manifest["case_route_visualization.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["coordinates", "routes", "total_route_distance"],
            }
        else:
            manifest["case_route_visualization.png"] = {"success": False, "reason": "缺少 seed=2026,c48,v4 的路线数据"}
    except Exception as exc:
        manifest["case_route_visualization.png"] = {"success": False, "reason": str(exc)}

    # 9) mode 箱线图
    try:
        if valid_rows:
            grp = defaultdict(list)
            for r in valid_rows:
                grp[_mode_name(str(r.get("mode", "")))].append(float(r.get("total_route_distance")))
            labels = sorted(grp.keys())
            data = [grp[k] for k in labels]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_title("不同求解模式总路线距离箱线图")
            ax.set_xlabel("求解模式")
            ax.set_ylabel("总路线距离")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_mode_comparison_boxplot.png")
            manifest["fig_mode_comparison_boxplot.png"] = {
                "success": True,
                "source": str(results_path),
                "fields": ["mode", "total_route_distance"],
            }
        else:
            manifest["fig_mode_comparison_boxplot.png"] = {"success": False, "reason": "缺少有效 distance"}
    except Exception as exc:
        manifest["fig_mode_comparison_boxplot.png"] = {"success": False, "reason": str(exc)}

    # 10) bitstring 能量分布
    try:
        if energy_rows:
            grp = defaultdict(list)
            for r in energy_rows:
                m = str(r.get("method", ""))
                e = _to_float(r.get("mean_energy"))
                if e is None:
                    continue
                grp[m].append(e)
            labels = sorted(grp.keys())
            vals = [_mean(grp[k]) for k in labels]
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.bar(labels, vals)
            has_quafu = any("quafu" in k for k in labels)
            title = "随机采样/经典模拟退火/Quafu 测量 bitstring 的 BQM 能量分布"
            if not has_quafu:
                title += "（当前 Quafu measured bitstrings 不可用）"
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("采样方法")
            ax.set_ylabel("平均 BQM 能量")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_bitstring_energy_distribution.png")
            manifest["fig_bitstring_energy_distribution.png"] = {
                "success": True,
                "source": str(energy_path),
                "fields": ["method", "mean_energy"],
            }
        else:
            manifest["fig_bitstring_energy_distribution.png"] = {"success": False, "reason": "缺少 energy_distribution.csv"}
    except Exception as exc:
        manifest["fig_bitstring_energy_distribution.png"] = {"success": False, "reason": str(exc)}

    # 11) top-k 对比
    try:
        if topk_rows:
            grp = defaultdict(list)
            for r in topk_rows:
                key = f"{r.get('method')}_k{r.get('k')}"
                val = _to_float(r.get("best_route_distance"))
                if val is None:
                    continue
                grp[key].append(val)
            labels = sorted(grp.keys())
            vals = [_mean(grp[k]) for k in labels]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(range(len(labels)), vals)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title("top-k 候选解细化后的路线距离对比")
            ax.set_xlabel("方法")
            ax.set_ylabel("最优路线距离（均值）")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_topk_candidate_refinement.png")
            manifest["fig_topk_candidate_refinement.png"] = {
                "success": True,
                "source": str(topk_path),
                "fields": ["method", "k", "best_route_distance"],
            }
        else:
            manifest["fig_topk_candidate_refinement.png"] = {"success": False, "reason": "缺少 topk_refinement.csv"}
    except Exception as exc:
        manifest["fig_topk_candidate_refinement.png"] = {"success": False, "reason": str(exc)}

    # 12) 精确差距图
    try:
        if exact_rows:
            grp = defaultdict(list)
            for r in exact_rows:
                method = str(r.get("method", ""))
                gap = _to_float(r.get("energy_gap"))
                if gap is None:
                    continue
                grp[method].append(gap)
            labels = sorted(grp.keys())
            vals = [_mean(grp[k]) for k in labels]
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.bar(labels, vals)
            ax.set_title("小规模实例能量差距（相对精确最优）")
            ax.set_xlabel("方法")
            ax.set_ylabel("能量差距")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, figdir / "fig_exact_gap_small_instances.png")
            manifest["fig_exact_gap_small_instances.png"] = {
                "success": True,
                "source": str(exact_path),
                "fields": ["method", "energy_gap"],
            }
        else:
            manifest["fig_exact_gap_small_instances.png"] = {"success": False, "reason": "缺少 exact_comparison.csv"}
    except Exception as exc:
        manifest["fig_exact_gap_small_instances.png"] = {"success": False, "reason": str(exc)}

    # 13) 消融对比图
    try:
        if ablation_rows:
            methods = []
            before_vals = []
            after_vals = []
            for r in ablation_rows:
                m = str(r.get("method", ""))
                b = _to_float(r.get("route_distance_before_refine"))
                a = _to_float(r.get("route_distance_after_refine"))
                if b is None or a is None:
                    continue
                methods.append(m)
                before_vals.append(b)
                after_vals.append(a)
            if methods:
                fig, ax = plt.subplots(figsize=(12, 5))
                xs = list(range(len(methods)))
                ax.plot(xs, before_vals, marker="o", label="细化前")
                ax.plot(xs, after_vals, marker="s", label="细化后")
                ax.set_xticks(xs)
                ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
                ax.set_title("不同 pipeline 的路线细化前后对比")
                ax.set_xlabel("pipeline")
                ax.set_ylabel("路线距离")
                ax.legend()
                ax.grid(alpha=0.25)
                _save(fig, figdir / "fig_pipeline_ablation.png")
                manifest["fig_pipeline_ablation.png"] = {
                    "success": True,
                    "source": str(ablation_path),
                    "fields": ["method", "route_distance_before_refine", "route_distance_after_refine"],
                }
            else:
                manifest["fig_pipeline_ablation.png"] = {"success": False, "reason": "消融数据缺少可绘制距离字段"}
        else:
            manifest["fig_pipeline_ablation.png"] = {"success": False, "reason": "缺少 pipeline_ablation.csv"}
    except Exception as exc:
        manifest["fig_pipeline_ablation.png"] = {"success": False, "reason": str(exc)}

    # 诊断摘要附注
    manifest["quafu_diagnostic_overview"] = {
        "success": bool(quafu_diag),
        "source": str(quafu_diag_path),
        "fields": [
            "current_quantum_mode_classification",
            "submitted_successfully",
            "task_id",
            "backend",
            "endpoint",
            "has_bitstrings",
        ],
        "content": quafu_diag if quafu_diag else "未找到或读取失败",
    }

    manifest_path = figdir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入完成: {manifest_path}")


if __name__ == "__main__":
    main()
