from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
NAVY = "#153E75"
CYAN = "#0891B2"
GREEN = "#059669"
ORANGE = "#D97706"
PURPLE = "#6D28D9"
RED = "#DC2626"
SLATE = "#64748B"
LIGHT = "#F1F5F9"
GRID = "#CBD5E1"
INK = "#172033"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.8,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, xy, width, height, title, subtitle, color, fill, tag=None):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.5, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(box)
    if tag:
        ax.text(x + 0.12, y + height - 0.13, tag, color=color, fontsize=8, fontweight="bold", va="top")
    ax.text(x + width / 2, y + height * 0.59, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", fontsize=8.1, color=SLATE)
    return box


def arrow(ax, start, end, color=SLATE, style="-"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=color, linestyle=style)
    )


def graphical_abstract():
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(0.2, 4.82, "DeepBlock: a hardware-in-the-loop quantum refinement system", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.2, 4.47, "The quantum processor generates candidates; classical logic preserves feasibility and evaluates the true CVRP objective.", fontsize=9.5, color=SLATE)

    boxes = [
        (0.2, 2.65, 2.00, 1.15, "Feasible CVRP seed", "40 customers\n4 vehicles", BLUE, "#EFF6FF", "CLASSICAL"),
        (2.45, 2.65, 1.90, 1.15, "Boundary pool", "16 customers\n3 overlapping blocks", CYAN, "#ECFEFF", "DECOMPOSE"),
        (4.60, 2.65, 1.90, 1.15, "Sparse 8-bit QUBO", "QUBO edges =\nphysical couplers", GREEN, "#ECFDF5", "CO-DESIGN"),
        (6.75, 2.65, 1.90, 1.15, "Baihua QAOA", "p=1, 4,096 shots\nzero SWAP", ORANGE, "#FFF7ED", "QUANTUM"),
        (8.90, 2.65, 1.90, 1.15, "Independent gate", "bottom-10%\nQUBO-energy mass", PURPLE, "#F5F3FF", "EVIDENCE"),
        (11.05, 2.65, 1.75, 1.15, "Accept", "capacity repair\ntrue route cost", BLUE, "#EFF6FF", "LOOP"),
    ]
    for b in boxes:
        add_box(ax, (b[0], b[1]), b[2], b[3], b[4], b[5], b[6], b[7], b[8])
    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (left[0] + left[2], 3.225), (right[0], 3.225))
    ax.add_patch(FancyArrowPatch((12.48, 2.65), (1.20, 2.65), connectionstyle="arc3,rad=-0.20", arrowstyle="-|>", mutation_scale=12, linewidth=1.25, linestyle="--", color=BLUE))

    cards = [
        (0.35, "2.73x", "hardware enrichment\nvs uniform sampling", PURPLE),
        (3.20, "+17.60 pp", "extra low-energy\nprobability mass", BLUE),
        (6.05, "6 / 6", "independent seeds\npositive", GREEN),
        (8.90, "36 / 36", "compiled with\nzero SWAP", ORANGE),
        (11.05, "144 / 144", "offline replay\nchecks passed", CYAN),
    ]
    for x, value, label, color in cards:
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.82, 1.10, boxstyle="round,pad=0.04,rounding_size=0.05", facecolor="#FFFFFF", edgecolor=GRID, linewidth=1.0))
        ax.text(x + 0.91, 1.25, value, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
        ax.text(x + 0.91, 0.82, label, ha="center", va="center", fontsize=8.1, color=SLATE)
    save(fig, "fig01_graphical_abstract")


def evidence_ladder():
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.55, "What the experiment establishes - and where the claim stops", fontsize=15, fontweight="bold", color=NAVY)
    levels = [
        (0.7, 0.55, 8.6, 1.05, "L1  Physical execution", "36 jobs, 147,456 shots, archived counts and task IDs", GREEN, True),
        (1.2, 1.80, 7.6, 1.05, "L2  Distributional quantum signal", "27.76% low-energy mass vs 10.16% uniform; p=0.03125", PURPLE, True),
        (1.7, 3.05, 6.6, 1.05, "L3  Closed-loop optimization", "strict route improvement on 6/6 selected instances", BLUE, True),
        (2.2, 4.30, 5.6, 1.05, "L4  End-to-end advantage", "not established; matched random remains statistically tied", SLATE, False),
    ]
    for x, y, w, h, title, subtitle, color, supported in levels:
        face = color + "18" if supported else "#F8FAFC"
        poly = Polygon([(x, y), (x + w, y), (x + w - 0.35, y + h), (x + 0.35, y + h)], closed=True, facecolor=face, edgecolor=color, linewidth=1.6)
        ax.add_patch(poly)
        ax.text(x + 0.55, y + 0.67, title, fontsize=11, fontweight="bold", color=color)
        ax.text(x + 0.55, y + 0.30, subtitle, fontsize=8.8, color=INK)
        ax.text(x + w - 0.48, y + 0.80, "SUPPORTED" if supported else "OPEN", ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)
    ax.text(5, 5.72, "Competition claim boundary", ha="center", fontsize=9.5, color=SLATE)
    ax.plot([2.2, 7.8], [5.53, 5.53], color=SLATE, linewidth=1.1, linestyle="--")
    ax.text(0.7, 0.18, "The paper wins credibility by separating a positive quantum kernel result from a speedup claim.", fontsize=9.5, color=SLATE)
    save(fig, "fig02_evidence_ladder")


def topology_codesign():
    qubits = [41, 28, 29, 30, 17, 16, 15, 14]
    fidelities = [0.987, 0.995, 0.986, 0.985, 0.985, 0.982, 0.994]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-0.7, 1.4)
    ax.axis("off")
    ax.set_title("Topology co-design: QUBO edges are physical couplers", loc="left", color=NAVY, fontweight="bold")
    for i in range(7):
        ax.plot([i, i + 1], [0, 0], color=CYAN, linewidth=5, solid_capstyle="round", zorder=1)
        ax.text(i + 0.5, 0.28, f"F={fidelities[i]:.3f}", ha="center", fontsize=8, color=SLATE)
    for i, q in enumerate(qubits):
        circ = plt.Circle((i, 0), 0.25, facecolor="#EFF6FF", edgecolor=BLUE, linewidth=1.8)
        circ.set_zorder(3)
        ax.add_patch(circ)
        ax.text(i, 0, str(q), ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY, zorder=4)
        ax.text(i, -0.43, f"x{i}", ha="center", fontsize=8.5, color=SLATE)
    ax.text(7.48, 0.86, "7 retained quadratic terms", ha="right", fontsize=9, color=GREEN, fontweight="bold")
    ax.text(7.48, 0.58, "0 routing SWAPs in all formal jobs", ha="right", fontsize=9, color=ORANGE, fontweight="bold")

    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    ax2.axis("off")
    stages = [
        (0.25, 0.95, 1.55, 1.05, "Prepare", r"$H^{\otimes 8}$", BLUE, "#EFF6FF"),
        (2.15, 0.95, 2.05, 1.05, "Cost phase", r"$R_Z(h_i\gamma)$ + $R_{ZZ}(J_{ij}\gamma)$", PURPLE, "#F5F3FF"),
        (4.55, 0.95, 1.65, 1.05, "Mixer", r"$R_X(2\beta)$", GREEN, "#ECFDF5"),
        (6.55, 0.95, 1.55, 1.05, "Measure", "8 classical bits", ORANGE, "#FFF7ED"),
        (8.45, 0.95, 1.25, 1.05, "Decode", "top-k", CYAN, "#ECFEFF"),
    ]
    for x, y, w, h, title, subtitle, color, fill in stages:
        add_box(ax2, (x, y), w, h, title, subtitle, color, fill)
    for a, b in zip(stages[:-1], stages[1:]):
        arrow(ax2, (a[0] + a[2], 1.475), (b[0], 1.475))
    ax2.text(0.25, 2.52, "Formal p=1 circuit anatomy", fontsize=11.5, fontweight="bold", color=NAVY)
    ax2.text(9.7, 0.35, "mean depth 27.7  |  mean two-qubit layers 13.3", ha="right", fontsize=8.8, color=SLATE)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig03_topology_codesign")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def seed_enrichment():
    rows = read_csv(ROOT / "results" / "baihua_quantum_candidate_quality_20260801" / "quantum_candidate_quality_seeds.csv")
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in rows]
    hw = np.array([float(r["hardware_low_energy_mass"]) * 100 for r in rows])
    ideal = np.array([float(r["ideal_low_energy_mass"]) * 100 for r in rows])
    uniform = np.array([float(r["uniform_low_energy_mass"]) * 100 for r in rows])
    x = np.arange(len(seeds))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.axhspan(0, uniform[0], color="#F8FAFC", zorder=0)
    ax.axhline(uniform[0], color=SLATE, linewidth=1.4, linestyle="--", label="uniform: 10.16%")
    for i in x:
        ax.plot([i, i], [uniform[i], hw[i]], color=PURPLE, linewidth=2.5, alpha=0.75)
    ax.scatter(x, hw, s=72, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4, label="Baihua hardware")
    ax.scatter(x, ideal, s=65, marker="D", facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=4, label="ideal QAOA")
    for i, value in enumerate(hw):
        ax.text(i, value + 1.25, f"{value:.1f}%", ha="center", fontsize=8.2, color=PURPLE, fontweight="bold")
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("Independent seed")
    ax.set_ylabel("Probability mass in bottom-10% energy region (%)")
    ax.set_ylim(0, 62)
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("All six seeds exceed the uniform baseline", loc="left", fontweight="bold", color=NAVY)

    ax2 = axes[1]
    means = [10.15625, hw.mean(), ideal.mean()]
    colors = [SLATE, PURPLE, BLUE]
    bars = ax2.bar([0, 1, 2], means, color=colors, width=0.64)
    ax2.set_xticks([0, 1, 2], ["Uniform", "Baihua", "Ideal"])
    ax2.set_ylim(0, 45)
    ax2.grid(axis="y", color=GRID, alpha=0.55)
    ax2.set_title("Mean low-energy mass", loc="left", fontweight="bold", color=NAVY)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax2.annotate("2.73x vs uniform", xy=(1, means[1]), xytext=(1.65, 33), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4), fontsize=9, color=PURPLE, fontweight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, "fig04_seed_enrichment")


def end_to_end():
    rows = read_csv(ROOT / "results" / "baihua_competition_package_20260801" / "formal_seed_results.csv")
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in rows]
    exact = np.array([float(r["exact_improvement"]) for r in rows])
    hw = np.array([float(r["baihua_improvement"]) for r in rows])
    rnd = np.array([float(r["random_improvement"]) for r in rows])
    sim = np.array([float(r["sim_improvement"]) for r in rows])
    x = np.arange(len(seeds))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    ax.bar(x, exact, width=0.72, color="#E2E8F0", edgecolor="#CBD5E1", label="exact local headroom")
    ax.scatter(x - 0.17, hw, s=58, marker="o", color=PURPLE, label="Baihua")
    ax.scatter(x, rnd, s=52, marker="s", facecolor="white", edgecolor=SLATE, linewidth=1.4, label="random")
    ax.scatter(x + 0.17, sim, s=55, marker="^", facecolor="white", edgecolor=BLUE, linewidth=1.4, label="ideal")
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("Independent seed")
    ax.set_ylabel("Route-distance improvement")
    ax.set_title("Closed-loop improvement within available local headroom", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax2 = axes[1]
    labels = ["Baihua", "Random", "Ideal", "Exact"]
    values = [hw.mean(), rnd.mean(), sim.mean(), exact.mean()]
    colors = [PURPLE, SLATE, BLUE, GREEN]
    bars = ax2.barh(np.arange(4), values, color=colors, height=0.62)
    ax2.set_yticks(np.arange(4), labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean route-distance improvement")
    ax2.set_title("Six-seed mean", loc="left", fontweight="bold", color=NAVY)
    ax2.grid(axis="x", color=GRID, alpha=0.55)
    for bar, val in zip(bars, values):
        ax2.text(val + 0.08, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8.8, fontweight="bold")
    ax2.text(0.02, -0.20, "Hardware: 6/6 positive\n71.4% mean exact-headroom recovery", transform=ax2.transAxes, fontsize=9, color=PURPLE, fontweight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, "fig05_end_to_end")


def candidate_budget():
    k = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    p1 = np.array([0.0, 8.3, 8.3, 12.5, 29.2, 33.3, 37.5, 41.7, 58.3])
    p2 = np.array([4.2, 4.2, 20.8, 25.0, 33.3, 41.7, 41.7, 50.0, 70.8])
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(k, p1, color=PURPLE, marker="o", linewidth=2.1, markersize=5.5, label="hardware p=1")
    ax.plot(k, p2, color=ORANGE, marker="s", linewidth=2.1, markersize=5.2, label="hardware p=2")
    ax.axvspan(8, 64, color="#DBEAFE", alpha=0.55)
    ax.annotate("formal decoding budget", xy=(64, 37.5), xytext=(34, 17), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3), color=BLUE, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_ylim(-2, 76)
    ax.set_xlabel("Number of full-objective candidates evaluated (k)")
    ax.set_ylabel("Subproblems containing an improving candidate (%)")
    ax.set_title("Useful states exist beyond the most frequent bitstrings", loc="left", fontweight="bold", color=NAVY)
    ax.grid(color=GRID, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.05, "Fixed-count retrospective diagnostic; not a new hardware run", transform=ax.transAxes, ha="right", fontsize=8.2, color=SLATE)
    fig.tight_layout()
    save(fig, "fig06_candidate_budget")


def ablation():
    labels = ["p1 k8", "p1 k16", "p1 k32", "p1 k64", "p2 k8", "p2 k32", "no overlap", "forward only"]
    sim = np.array([1.054, 1.468, 1.321, 1.733, 0.675, 1.407, 1.187, 1.226])
    rnd = np.array([0.917, 1.159, 1.520, 1.654, 0.917, 1.520, 1.460, 1.270])
    delta = sim - rnd
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.barh(y + 0.18, sim, height=0.34, color=BLUE, label="ideal QAOA")
    ax.barh(y - 0.18, rnd, height=0.34, color=SLATE, label="matched random")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean improvement over 10 paired seeds")
    ax.set_title("Configuration screen", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", color=GRID, alpha=0.55)
    ax.legend(frameon=False, loc="lower right")
    ax.axhspan(2.5, 3.5, color="#DBEAFE", alpha=0.45)
    ax.text(1.70, 3.1, "selected", fontsize=8.5, color=BLUE, fontweight="bold", ha="right")

    ax2 = axes[1]
    colors = [GREEN if d >= 0 else "#CBD5E1" for d in delta]
    ax2.barh(y, delta, color=colors, height=0.56)
    ax2.axvline(0, color=SLATE, linewidth=1)
    ax2.set_yticks(y, ["" for _ in y])
    ax2.invert_yaxis()
    ax2.set_xlabel("QAOA minus random")
    ax2.set_title("Paired contrast", loc="left", fontweight="bold", color=NAVY)
    ax2.grid(axis="x", color=GRID, alpha=0.55)
    for yi, d in zip(y, delta):
        ax2.text(d + (0.015 if d >= 0 else -0.015), yi, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=8.2, color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig07_ablation")


def reproducibility_chain():
    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(0.2, 3.82, "Audit trail from cloud task to scientific claim", fontsize=15, fontweight="bold", color=NAVY)
    items = [
        (0.25, "Calibration", "timestamp\nqubit metrics", BLUE),
        (2.10, "Circuit", "logical + physical\nQASM", PURPLE),
        (3.95, "Cloud job", "task ID\n4,096-shot counts", ORANGE),
        (5.80, "Mapping", "physical qubits\n7 couplers", CYAN),
        (7.65, "Replay", "candidate repair\nstate transitions", GREEN),
        (9.50, "Decision", "quality gate\npaired statistics", NAVY),
    ]
    for x, title, subtitle, color in items:
        add_box(ax, (x, 1.45), 1.55, 1.28, title, subtitle, color, color + "12")
    for a, b in zip(items[:-1], items[1:]):
        arrow(ax, (a[0] + 1.55, 2.09), (b[0], 2.09))
    ax.text(0.25, 0.70, "36 hardware jobs", fontsize=10, fontweight="bold", color=ORANGE)
    ax.text(2.70, 0.70, "147,456 shots", fontsize=10, fontweight="bold", color=PURPLE)
    ax.text(5.05, 0.70, "0 SWAP", fontsize=10, fontweight="bold", color=CYAN)
    ax.text(6.95, 0.70, "144 / 144 replay checks", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(10.95, 0.70, "falsifiable", fontsize=10, fontweight="bold", color=NAVY, ha="right")
    ax.plot([0.25, 11.05], [0.48, 0.48], color=GRID, linewidth=1)
    ax.text(0.25, 0.16, "Raw observations, transformations and acceptance decisions are separated so a judge can independently reconstruct the result.", fontsize=8.8, color=SLATE)
    save(fig, "fig08_reproducibility_chain")


def main():
    graphical_abstract()
    evidence_ladder()
    topology_codesign()
    seed_enrichment()
    end_to_end()
    candidate_budget()
    ablation()
    reproducibility_chain()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
