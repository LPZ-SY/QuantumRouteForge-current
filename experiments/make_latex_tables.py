from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _tex_table(caption: str, label: str, headers: list[str], rows: list[list[str]]) -> str:
    cols = "l" * len(headers)
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    if rows:
        for r in rows:
            lines.append(" & ".join(r) + " \\\\")
    else:
        lines.append("TODO & TODO & TODO \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 LaTeX 中文表格")
    parser.add_argument("--results", type=str, default="results/competition_runs/results.csv")
    parser.add_argument("--outdir", type=str, default="paper/tables")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comp_rows = _read_csv(ROOT / args.results)
    energy_rows = _read_csv(ROOT / "results/energy_distribution/energy_distribution.csv")
    topk_rows = _read_csv(ROOT / "results/topk_refinement/topk_refinement.csv")
    exact_rows = _read_csv(ROOT / "results/exact_runs/exact_comparison.csv")
    ablation_rows = _read_csv(ROOT / "results/pipeline_ablation/pipeline_ablation.csv")

    quafu_diag = {}
    diag_path = ROOT / "results/quafu_diagnostics/diagnostic_summary.json"
    if diag_path.exists():
        try:
            quafu_diag = json.loads(diag_path.read_text(encoding="utf-8"))
        except Exception:
            quafu_diag = {}

    # 1) experiment_summary
    exp_data = defaultdict(lambda: {"dist": [], "energy": [], "feasible": []})
    for r in comp_rows:
        mode = str(r.get("mode", ""))
        d = _to_float(r.get("total_route_distance"))
        e = _to_float(r.get("bqm_energy"))
        if d is not None:
            exp_data[mode]["dist"].append(d)
        if e is not None:
            exp_data[mode]["energy"].append(e)
        exp_data[mode]["feasible"].append(1.0 if _to_bool(r.get("feasible")) else 0.0)

    exp_rows = []
    for mode in sorted(exp_data.keys()):
        d = exp_data[mode]
        exp_rows.append(
            [
                mode,
                str(len(d["feasible"])),
                f"{_mean(d['dist']):.4f}" if d["dist"] else "TODO",
                f"{_mean(d['energy']):.4f}" if d["energy"] else "TODO",
                f"{_mean(d['feasible']):.4f}" if d["feasible"] else "TODO",
            ]
        )

    (outdir / "experiment_summary.tex").write_text(
        _tex_table(
            "总体实验结果汇总",
            "tab:experiment_summary",
            ["求解模式", "运行次数", "平均总路线距离", "平均BQM能量", "可行解比例"],
            exp_rows,
        ),
        encoding="utf-8",
    )

    # 2) quafu_status_summary
    q_status = defaultdict(int)
    for r in comp_rows:
        if str(r.get("mode", "")) == "quantum":
            q_status[str(r.get("quafu_status", "unknown"))] += 1

    q_rows = [[k, str(v)] for k, v in sorted(q_status.items(), key=lambda kv: kv[0])]
    if quafu_diag:
        q_rows.append(["diagnostic_classification", str(quafu_diag.get("current_quantum_mode_classification", "unknown"))])

    (outdir / "quafu_status_summary.tex").write_text(
        _tex_table(
            "Quafu 任务状态统计",
            "tab:quafu_status_summary",
            ["状态", "次数/结果"],
            q_rows,
        ),
        encoding="utf-8",
    )

    # 3) scaling_summary
    sc = defaultdict(lambda: {"runtime": [], "vars": [], "quads": []})
    for r in comp_rows:
        c = _to_int(r.get("customers"))
        rt = _to_float(r.get("runtime_total_sec"))
        nv = _to_float(r.get("num_bqm_variables"))
        nq = _to_float(r.get("num_bqm_quadratic_terms"))
        if c is None:
            continue
        if rt is not None:
            sc[c]["runtime"].append(rt)
        if nv is not None:
            sc[c]["vars"].append(nv)
        if nq is not None:
            sc[c]["quads"].append(nq)

    sc_rows = []
    for c in sorted(sc.keys()):
        d = sc[c]
        sc_rows.append(
            [
                str(c),
                f"{_mean(d['runtime']):.4f}" if d["runtime"] else "TODO",
                f"{_mean(d['vars']):.2f}" if d["vars"] else "TODO",
                f"{_mean(d['quads']):.2f}" if d["quads"] else "TODO",
            ]
        )

    (outdir / "scaling_summary.tex").write_text(
        _tex_table(
            "规模扩展统计",
            "tab:scaling_summary",
            ["客户数量", "平均运行时间(秒)", "平均BQM变量数", "平均BQM二次项数"],
            sc_rows,
        ),
        encoding="utf-8",
    )

    # 4) energy_distribution_summary
    ed = defaultdict(lambda: {"best": [], "mean": [], "median": [], "feasible": []})
    for r in energy_rows:
        m = str(r.get("method", ""))
        b = _to_float(r.get("best_energy"))
        me = _to_float(r.get("mean_energy"))
        md = _to_float(r.get("median_energy"))
        fr = _to_float(r.get("feasible_sample_rate"))
        if b is not None:
            ed[m]["best"].append(b)
        if me is not None:
            ed[m]["mean"].append(me)
        if md is not None:
            ed[m]["median"].append(md)
        if fr is not None:
            ed[m]["feasible"].append(fr)

    ed_rows = []
    for m in sorted(ed.keys()):
        d = ed[m]
        ed_rows.append(
            [
                m,
                f"{_mean(d['best']):.4f}" if d["best"] else "TODO",
                f"{_mean(d['mean']):.4f}" if d["mean"] else "TODO",
                f"{_mean(d['median']):.4f}" if d["median"] else "TODO",
                f"{_mean(d['feasible']):.4f}" if d["feasible"] else "TODO",
            ]
        )

    (outdir / "energy_distribution_summary.tex").write_text(
        _tex_table(
            "BQM 能量分布汇总",
            "tab:energy_distribution_summary",
            ["方法", "最优能量均值", "平均能量均值", "中位能量均值", "可行样本比例均值"],
            ed_rows,
        ),
        encoding="utf-8",
    )

    # 5) exact_gap_summary
    eg = defaultdict(lambda: {"gap": [], "ratio": []})
    for r in exact_rows:
        m = str(r.get("method", ""))
        g = _to_float(r.get("energy_gap"))
        ar = _to_float(r.get("approximation_ratio"))
        if g is not None:
            eg[m]["gap"].append(g)
        if ar is not None:
            eg[m]["ratio"].append(ar)

    eg_rows = []
    for m in sorted(eg.keys()):
        d = eg[m]
        eg_rows.append(
            [
                m,
                f"{_mean(d['gap']):.4f}" if d["gap"] else "TODO",
                f"{_mean(d['ratio']):.4f}" if d["ratio"] else "TODO",
            ]
        )

    (outdir / "exact_gap_summary.tex").write_text(
        _tex_table(
            "小规模精确对照能量差距",
            "tab:exact_gap_summary",
            ["方法", "平均能量差距", "平均近似比"],
            eg_rows,
        ),
        encoding="utf-8",
    )

    # 6) pipeline_ablation_summary
    ab = defaultdict(lambda: {"before": [], "after": [], "impr": []})
    for r in ablation_rows:
        m = str(r.get("method", ""))
        b = _to_float(r.get("route_distance_before_refine"))
        a = _to_float(r.get("route_distance_after_refine"))
        i = _to_float(r.get("improvement_percent"))
        if b is not None:
            ab[m]["before"].append(b)
        if a is not None:
            ab[m]["after"].append(a)
        if i is not None:
            ab[m]["impr"].append(i)

    ab_rows = []
    for m in sorted(ab.keys()):
        d = ab[m]
        ab_rows.append(
            [
                m,
                f"{_mean(d['before']):.4f}" if d["before"] else "TODO",
                f"{_mean(d['after']):.4f}" if d["after"] else "TODO",
                f"{_mean(d['impr']):.4f}" if d["impr"] else "TODO",
            ]
        )

    (outdir / "pipeline_ablation_summary.tex").write_text(
        _tex_table(
            "优化流程消融实验汇总",
            "tab:pipeline_ablation_summary",
            ["流程方法", "细化前距离均值", "细化后距离均值", "改进比例均值(%)"],
            ab_rows,
        ),
        encoding="utf-8",
    )

    # 可选：topk 简要附注
    if topk_rows:
        (outdir / "topk_refinement_note.tex").write_text(
            "% top-k 实验已生成 CSV，可在正文中引用结果图。\n",
            encoding="utf-8",
        )

    print(f"写入完成: {outdir}")


if __name__ == "__main__":
    main()
