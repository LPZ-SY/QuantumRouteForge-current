from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization  # noqa: E402
from quantum_route_forge.env import quafu_token as load_quafu_token  # noqa: E402


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#17becf",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

PAGE_STYLE = {
    "maxWidth": "1400px",
    "margin": "0 auto",
    "padding": "20px",
    "fontFamily": "Segoe UI, Microsoft YaHei, sans-serif",
    "color": "#0f1d2d",
    "background": "linear-gradient(180deg, #f4f8ff 0%, #fbfdff 100%)",
}

PANEL_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #dde7f3",
    "borderRadius": "14px",
    "padding": "14px",
    "boxShadow": "0 3px 14px rgba(39, 84, 126, 0.08)",
}

GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))",
    "gap": "12px",
}

FIELD_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "6px",
}

LABEL_STYLE = {
    "fontSize": "13px",
    "fontWeight": "700",
    "letterSpacing": "0.2px",
    "color": "#2a4159",
}

INPUT_STYLE = {
    "width": "100%",
    "height": "40px",
    "borderRadius": "10px",
}

HINT_STYLE = {
    "fontSize": "11px",
    "lineHeight": "1.2",
    "color": "#6582a0",
}

BUTTON_STYLE = {
    "width": "100%",
    "height": "42px",
    "border": "none",
    "borderRadius": "10px",
    "background": "linear-gradient(90deg, #0d63ce 0%, #1f8bff 100%)",
    "color": "white",
    "fontWeight": "700",
    "cursor": "pointer",
}

STATUS_BOX_STYLE = {
    "marginBottom": "8px",
    "fontWeight": "600",
    "wordBreak": "break-word",
    "lineHeight": "1.45",
    "backgroundColor": "#f7fbff",
    "border": "1px solid #d8e8fa",
    "borderRadius": "10px",
    "padding": "10px 12px",
}

METRICS_BOX_STYLE = {
    "fontWeight": "600",
    "marginBottom": "8px",
    "backgroundColor": "#f7fbff",
    "border": "1px solid #d8e8fa",
    "borderRadius": "10px",
    "padding": "10px 12px",
}


def _field(label: str, control, hint: str = "", span: int = 1):
    style = dict(FIELD_STYLE)
    if span > 1:
        style["gridColumn"] = f"span {span}"
    children = [html.Label(label, style=LABEL_STYLE), control]
    if hint:
        children.append(html.Div(hint, style=HINT_STYLE))
    return html.Div(children, style=style)


def _build_figure(result) -> go.Figure:
    depot = result.instance.depot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[depot[0]],
            y=[depot[1]],
            mode="markers",
            name="Depot",
            marker={"size": 18, "symbol": "star", "color": "#111111"},
        )
    )

    for idx, route in enumerate(result.routes):
        color = COLORS[idx % len(COLORS)]
        xs = [depot[0]] + [c.x for c in route.customers] + [depot[0]]
        ys = [depot[1]] + [c.y for c in route.customers] + [depot[1]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"Vehicle {route.vehicle_id}",
                line={"width": 3, "color": color},
                marker={"size": 8},
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Quantum Route Forge",
        xaxis_title="City X",
        yaxis_title="City Y",
        height=700,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title="Fleet",
    )
    return fig


def _build_result_table(result) -> html.Table:
    th_style = {
        "textAlign": "left",
        "padding": "10px 12px",
        "borderBottom": "1px solid #d6e4f2",
        "backgroundColor": "#eef5fd",
        "fontWeight": "700",
        "fontSize": "13px",
        "color": "#223b55",
    }
    td_style = {
        "padding": "9px 12px",
        "borderBottom": "1px solid #e7eef7",
        "fontSize": "13px",
        "color": "#1d3147",
        "verticalAlign": "top",
    }
    rows = []
    for route in result.routes:
        stop_ids = ", ".join(str(c.customer_id) for c in route.customers[:14])
        if len(route.customers) > 14:
            stop_ids += " ..."
        rows.append(
            html.Tr(
                [
                    html.Td(f"Vehicle {route.vehicle_id}", style=td_style),
                    html.Td(str(len(route.customers)), style=td_style),
                    html.Td(str(route.load), style=td_style),
                    html.Td(f"{route.distance:.2f}", style=td_style),
                    html.Td(stop_ids, style=td_style),
                ]
            )
        )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Route", style=th_style),
                        html.Th("Stops", style=th_style),
                        html.Th("Load", style=th_style),
                        html.Th("Distance", style=th_style),
                        html.Th("Customer IDs", style=th_style),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={
            "width": "100%",
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "overflow": "hidden",
            "borderRadius": "10px",
            "border": "1px solid #d6e4f2",
            "backgroundColor": "white",
        },
    )


def _build_error_outputs(instance, detail: str):
    min_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
    status = f"Input error: {detail}"
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title="No Feasible Solution",
        height=480,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Capacity infeasible under strict mode.<br>"
                f"Demand={instance.total_demand}, vehicles={instance.num_vehicles}, "
                f"required capacity >= {min_capacity}.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15, "color": "#8a1f1f"},
            }
        ],
    )
    metrics = (
        f"Total demand={instance.total_demand}, fleet capacity={instance.num_vehicles * instance.vehicle_capacity} "
        f"(required >= {instance.num_vehicles * min_capacity})."
    )
    table = html.Div(
        "No route generated. Increase Capacity or Vehicles, then run again.",
        style={"fontWeight": "600", "color": "#8a1f1f", "padding": "8px"},
    )
    return status, fig, metrics, table


def _generate_outputs(
    seed,
    customers,
    vehicles,
    capacity,
    mode,
    time_limit,
    quafu_token,
    quafu_backend,
    quafu_base_url,
    quafu_shots,
    quafu_max_qubits,
    quafu_wait,
    quafu_timeout_sec,
    quafu_proxy_url,
    quafu_verify_ssl,
    quafu_result_task_id,
    quafu_manual_bitstring,
    tabu_iterations=20,
    tabu_tenure=5,
    qaoa_subproblem_size=6,
    qaoa_max_edges=10,
    qaoa_gamma=1.1,
    qaoa_beta=0.8,
    routing_method="ortools",
):
    seed = int(seed or 2026)
    customers = max(8, int(customers or 48))
    vehicles = max(1, int(vehicles or 4))
    capacity = max(1, int(capacity or 34))
    time_limit = max(5, int(time_limit or 10))
    quafu_token = (quafu_token or "").strip() or load_quafu_token()
    quafu_backend = (quafu_backend or "").strip()
    quafu_base_url = (quafu_base_url or "").strip() or os.getenv("QUAFU_BASE_URL", "")
    quafu_shots = max(100, int(quafu_shots or 1024))
    quafu_max_qubits = max(2, min(6, int(quafu_max_qubits or 6)))
    quafu_wait = str(quafu_wait).lower().strip() == "true"
    quafu_timeout_sec = max(5, int(quafu_timeout_sec or os.getenv("QUAFU_TIMEOUT_SEC", "25")))
    quafu_proxy_url = (quafu_proxy_url or "").strip() or os.getenv("QUAFU_PROXY_URL", "")
    quafu_verify_ssl = str(quafu_verify_ssl).lower().strip() == "true"
    quafu_result_task_id = (quafu_result_task_id or "").strip()
    quafu_manual_bitstring = (quafu_manual_bitstring or "").strip()
    tabu_iterations = max(0, min(100, int(tabu_iterations or 20)))
    tabu_tenure = max(1, min(50, int(tabu_tenure or 5)))
    qaoa_subproblem_size = max(2, min(6, int(qaoa_subproblem_size or 6)))
    qaoa_max_edges = max(1, min(10, int(qaoa_max_edges or 10)))
    qaoa_gamma = float(qaoa_gamma if qaoa_gamma is not None else 1.1)
    qaoa_beta = float(qaoa_beta if qaoa_beta is not None else 0.8)
    routing_method = (routing_method or "ortools").lower().strip()

    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )

    if not instance.feasible_capacity:
        min_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
        return _build_error_outputs(
            instance,
            (
                "Total demand exceeds fleet capacity. "
                f"Set capacity >= {min_capacity} or increase vehicles."
            ),
        )

    try:
        result = run_optimization(
            instance=instance,
            mode=mode,
            time_limit=time_limit,
            num_reads=120,
            quafu_token=quafu_token,
            quafu_backend=quafu_backend,
            quafu_base_url=quafu_base_url,
            quafu_shots=quafu_shots,
            quafu_wait=quafu_wait,
            quafu_max_qubits=quafu_max_qubits,
            quafu_timeout_sec=quafu_timeout_sec,
            quafu_proxy_url=quafu_proxy_url,
            quafu_verify_ssl=quafu_verify_ssl,
            quafu_result_task_id=quafu_result_task_id,
            quafu_manual_bitstring=quafu_manual_bitstring,
            auto_repair_capacity=False,
            tabu_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            qaoa_subproblem_size=qaoa_subproblem_size,
            qaoa_max_edges=qaoa_max_edges,
            qaoa_gamma=qaoa_gamma,
            qaoa_beta=qaoa_beta,
            clustering_seed=seed,
            routing_method=routing_method,
        )
    except ValueError as exc:
        return _build_error_outputs(instance, str(exc))

    fig = _build_figure(result)
    used = result.metadata.used_mode
    msg = result.metadata.message
    qmeta = []
    if result.metadata.quantum_backend:
        qmeta.append(f"backend={result.metadata.quantum_backend}")
    if result.metadata.quantum_task_id:
        qmeta.append(f"task_id={result.metadata.quantum_task_id}")
    if result.metadata.quantum_bitstring:
        qmeta.append(f"seed_bitstring={result.metadata.quantum_bitstring}")
    if result.metadata.quantum_endpoint:
        qmeta.append(f"endpoint={result.metadata.quantum_endpoint}")
    qmeta_text = f" | Quafu: {', '.join(qmeta)}" if qmeta else ""
    objective_label = (
        "Route proxy"
        if "tabu_qaoa" in used
        else "BQM energy"
    )
    status = (
        f"Requested solver: {mode} | Used: {used} | "
        f"{objective_label}: {result.metadata.energy:.3f}{qmeta_text} | {msg}"
    )
    total_load = sum(r.load for r in result.routes)
    ablation = result.metadata.stratified_ablation or {}
    layers = ablation.get("layers", {})
    layer_text = ", ".join(
        f"k={sub_k}: N={layer.get('iterations', 0)}"
        for sub_k, layer in sorted(
            layers.items(),
            key=lambda item: int(item[0]),
        )
    )
    layer_suffix = f", QAOA strata [{layer_text}]" if layer_text else ""
    metrics = (
        f"Total demand={instance.total_demand}, served load={total_load}, "
        f"fleet capacity={instance.num_vehicles * instance.vehicle_capacity}, "
        f"total route distance={result.total_distance:.2f}{layer_suffix}"
    )
    table = _build_result_table(result)
    return status, fig, metrics, table


app = Dash(__name__)
app.title = "Quantum Route Forge"

_initial_status, _initial_figure, _initial_metrics, _initial_table = _generate_outputs(
    seed=2026,
    customers=18,
    vehicles=4,
    capacity=28,
    mode="classical",
    time_limit=8,
    quafu_token="",
    quafu_backend="",
    quafu_base_url="",
    quafu_shots=1024,
    quafu_max_qubits=6,
    quafu_wait="false",
    quafu_timeout_sec=25,
    quafu_proxy_url="",
    quafu_verify_ssl="true",
    quafu_result_task_id="",
    quafu_manual_bitstring="",
    tabu_iterations=20,
    tabu_tenure=5,
    qaoa_subproblem_size=6,
    qaoa_max_edges=10,
    qaoa_gamma=1.1,
    qaoa_beta=0.8,
    routing_method="ortools",
)

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.Div(
            [
                html.H1(
                    "Quantum Route Forge",
                    style={"marginBottom": "8px", "fontSize": "44px", "lineHeight": "1.05"},
                ),
                html.P(
                    (
                        "Capacity clustering + sparse Max-Cut Tabu-QAOA "
                        "+ capacity repair + OR-Tools routing."
                    ),
                    style={"marginTop": "0", "color": "#425d78", "fontSize": "18px"},
                ),
            ],
            style={"padding": "4px 4px 12px 4px"},
        ),
        html.Div(
            style={**PANEL_STYLE, "marginBottom": "12px"},
            children=[
                html.Div(
                    "Hybrid Optimizer",
                    style={"fontWeight": "700", "fontSize": "15px", "marginBottom": "10px", "color": "#1a3552"},
                ),
                html.Div(
                    style=GRID_STYLE,
                    children=[
                        _field(
                            "Tabu Iterations",
                            dcc.Input(id="tabu-iterations", type="number", min=0, max=100, value=20, style=INPUT_STYLE),
                        ),
                        _field(
                            "Tabu Tenure",
                            dcc.Input(id="tabu-tenure", type="number", min=1, max=50, value=5, style=INPUT_STYLE),
                        ),
                        _field(
                            "QAOA Customers",
                            dcc.Input(id="qaoa-subproblem-size", type="number", min=2, max=6, value=6, style=INPUT_STYLE),
                            "Maximum/target window size; the hardware limit is 6 qubits.",
                        ),
                        _field(
                            "Max-Cut Edges",
                            dcc.Input(id="qaoa-max-edges", type="number", min=1, max=10, value=10, style=INPUT_STYLE),
                            "Each edge becomes two CNOT gates; 10 edges means 20 CNOTs.",
                        ),
                        _field(
                            "QAOA Gamma",
                            dcc.Input(id="qaoa-gamma", type="number", value=1.1, step=0.05, style=INPUT_STYLE),
                        ),
                        _field(
                            "QAOA Beta",
                            dcc.Input(id="qaoa-beta", type="number", value=0.8, step=0.05, style=INPUT_STYLE),
                        ),
                        _field(
                            "Route Solver",
                            dcc.Dropdown(
                                id="routing-method",
                                options=[
                                    {"label": "OR-Tools TSP", "value": "ortools"},
                                    {"label": "Nearest Neighbor + 2-opt", "value": "heuristic"},
                                ],
                                value="ortools",
                                clearable=False,
                                style=INPUT_STYLE,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={**PANEL_STYLE, "marginBottom": "12px"},
            children=[
                html.Div(
                    "Scenario Settings",
                    style={"fontWeight": "700", "fontSize": "15px", "marginBottom": "10px", "color": "#1a3552"},
                ),
                html.Div(
                    style=GRID_STYLE,
                    children=[
                        _field("Seed", dcc.Input(id="seed", type="number", value=2026, style=INPUT_STYLE)),
                        _field(
                            "Customers",
                            dcc.Input(id="customers", type="number", min=8, max=160, value=48, style=INPUT_STYLE),
                        ),
                        _field(
                            "Vehicles",
                            dcc.Input(id="vehicles", type="number", min=1, max=8, value=4, style=INPUT_STYLE),
                        ),
                        _field(
                            "Capacity",
                            dcc.Input(id="capacity", type="number", min=5, max=120, value=34, style=INPUT_STYLE),
                        ),
                        _field(
                            "Mode",
                            dcc.Dropdown(
                                id="mode",
                                options=[
                                    {"label": "Quafu Real Quantum", "value": "quantum"},
                                    {
                                        "label": "Local Exact Max-Cut (no hardware)",
                                        "value": "hybrid_local",
                                    },
                                    {
                                        "label": "Legacy BQM Baseline",
                                        "value": "classical",
                                    },
                                ],
                                value="quantum",
                                clearable=False,
                                style=INPUT_STYLE,
                            ),
                        ),
                        _field(
                            "Time Limit (s)",
                            dcc.Input(id="time-limit", type="number", min=5, max=120, value=10, style=INPUT_STYLE),
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={**PANEL_STYLE, "marginBottom": "12px"},
            children=[
                html.Div(
                    "Quantum Connection",
                    style={"fontWeight": "700", "fontSize": "15px", "marginBottom": "10px", "color": "#1a3552"},
                ),
                html.Div(
                    style=GRID_STYLE,
                    children=[
                        _field(
                            "Token",
                            dcc.Input(
                                id="quafu-token",
                                type="password",
                                placeholder="QuarkStudio/Quafu token (optional when .env exists)",
                                style=INPUT_STYLE,
                            ),
                            "UI value overrides QUAFU_API_TOKEN; otherwise .env is loaded automatically.",
                            span=2,
                        ),
                        _field("Backend", dcc.Input(id="quafu-backend", type="text", placeholder="Dongling", style=INPUT_STYLE)),
                        _field(
                            "Base URL",
                            dcc.Input(
                                id="quafu-base-url",
                                type="text",
                                placeholder="https://quafu-sqc.baqis.ac.cn/",
                                style=INPUT_STYLE,
                            ),
                            span=2,
                        ),
                        _field(
                            "Shots",
                            dcc.Input(
                                id="quafu-shots",
                                type="number",
                                min=1024,
                                max=20480,
                                step=1024,
                                value=1024,
                                style=INPUT_STYLE,
                            ),
                            "QuarkStudio jobs use multiples of 1024 shots.",
                        ),
                        _field(
                            "Max Qubits",
                            dcc.Input(id="quafu-max-qubits", type="number", min=2, max=6, value=6, style=INPUT_STYLE),
                            "Hard limit is 6 for the sparse Max-Cut subproblem.",
                        ),
                        _field(
                            "Timeout (s)",
                            dcc.Input(
                                id="quafu-timeout-sec",
                                type="number",
                                min=5,
                                max=120,
                                value=25,
                                style=INPUT_STYLE,
                            ),
                        ),
                        _field(
                            "Proxy URL",
                            dcc.Input(
                                id="quafu-proxy-url",
                                type="text",
                                value="",
                                placeholder="Optional: leave empty unless you use local proxy",
                                style=INPUT_STYLE,
                            ),
                        ),
                        _field(
                            "Manual Seed Bitstring",
                            dcc.Input(
                                id="quafu-manual-bitstring",
                                type="text",
                                value="",
                                placeholder="Optional override, e.g. 010110",
                                style=INPUT_STYLE,
                            ),
                            "Use only 0/1. When provided, this seed overrides pending/missing Quafu bitstring.",
                            span=2,
                        ),
                        _field(
                            "Result Task ID",
                            dcc.Input(
                                id="quafu-result-task-id",
                                type="text",
                                value="",
                                placeholder="Optional: query existing task result first",
                                style=INPUT_STYLE,
                            ),
                            "When provided, the app first tries to fetch this task's bitstring before creating a new task.",
                            span=2,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={**PANEL_STYLE, "marginBottom": "12px"},
            children=[
                html.Div(
                    "Execution",
                    style={"fontWeight": "700", "fontSize": "15px", "marginBottom": "10px", "color": "#1a3552"},
                ),
                html.Div(
                    style=GRID_STYLE,
                    children=[
                        _field(
                            "Wait For Quantum Result",
                            dcc.Dropdown(
                                id="quafu-wait",
                                options=[
                                    {"label": "Yes (sync)", "value": "true"},
                                    {"label": "No (submit only)", "value": "false"},
                                ],
                                value="true",
                                clearable=False,
                                style=INPUT_STYLE,
                            ),
                        ),
                        _field(
                            "SSL Verify",
                            dcc.Dropdown(
                                id="quafu-verify-ssl",
                                options=[
                                    {"label": "On", "value": "true"},
                                    {"label": "Off (diagnostic)", "value": "false"},
                                ],
                                value="true",
                                clearable=False,
                                style=INPUT_STYLE,
                            ),
                        ),
                        _field(
                            "Run Optimization",
                            html.Button("Optimize", id="run-btn", n_clicks=0, style=BUTTON_STYLE),
                            "Runs assignment + route refinement.",
                        ),
                    ],
                ),
            ],
        ),
        dcc.Loading(
            type="dot",
            children=[
                html.Div(
                    id="status-line",
                    children=_initial_status,
                    style=STATUS_BOX_STYLE,
                ),
                html.Div(
                    [dcc.Graph(id="route-graph", figure=_initial_figure, config={"displayModeBar": False})],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    id="metrics-line",
                    children=_initial_metrics,
                    style=METRICS_BOX_STYLE,
                ),
                html.Div(id="table-area", children=_initial_table, style=PANEL_STYLE),
            ],
        ),
    ],
)


@app.callback(
    Output("status-line", "children"),
    Output("route-graph", "figure"),
    Output("metrics-line", "children"),
    Output("table-area", "children"),
    Input("run-btn", "n_clicks"),
    State("seed", "value"),
    State("customers", "value"),
    State("vehicles", "value"),
    State("capacity", "value"),
    State("mode", "value"),
    State("time-limit", "value"),
    State("quafu-token", "value"),
    State("quafu-backend", "value"),
    State("quafu-base-url", "value"),
    State("quafu-shots", "value"),
    State("quafu-max-qubits", "value"),
    State("quafu-wait", "value"),
    State("quafu-timeout-sec", "value"),
    State("quafu-proxy-url", "value"),
    State("quafu-verify-ssl", "value"),
    State("quafu-result-task-id", "value"),
    State("quafu-manual-bitstring", "value"),
    State("tabu-iterations", "value"),
    State("tabu-tenure", "value"),
    State("qaoa-subproblem-size", "value"),
    State("qaoa-max-edges", "value"),
    State("qaoa-gamma", "value"),
    State("qaoa-beta", "value"),
    State("routing-method", "value"),
)
def run_pipeline(
    _clicks,
    seed,
    customers,
    vehicles,
    capacity,
    mode,
    time_limit,
    quafu_token,
    quafu_backend,
    quafu_base_url,
    quafu_shots,
    quafu_max_qubits,
    quafu_wait,
    quafu_timeout_sec,
    quafu_proxy_url,
    quafu_verify_ssl,
    quafu_result_task_id,
    quafu_manual_bitstring,
    tabu_iterations,
    tabu_tenure,
    qaoa_subproblem_size,
    qaoa_max_edges,
    qaoa_gamma,
    qaoa_beta,
    routing_method,
):
    return _generate_outputs(
        seed=seed,
        customers=customers,
        vehicles=vehicles,
        capacity=capacity,
        mode=mode,
        time_limit=time_limit,
        quafu_token=quafu_token,
        quafu_backend=quafu_backend,
        quafu_base_url=quafu_base_url,
        quafu_shots=quafu_shots,
        quafu_max_qubits=quafu_max_qubits,
        quafu_wait=quafu_wait,
        quafu_timeout_sec=quafu_timeout_sec,
        quafu_proxy_url=quafu_proxy_url,
        quafu_verify_ssl=quafu_verify_ssl,
        quafu_result_task_id=quafu_result_task_id,
        quafu_manual_bitstring=quafu_manual_bitstring,
        tabu_iterations=tabu_iterations,
        tabu_tenure=tabu_tenure,
        qaoa_subproblem_size=qaoa_subproblem_size,
        qaoa_max_edges=qaoa_max_edges,
        qaoa_gamma=qaoa_gamma,
        qaoa_beta=qaoa_beta,
        routing_method=routing_method,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quantum Route Forge web app.")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()
    app.run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
