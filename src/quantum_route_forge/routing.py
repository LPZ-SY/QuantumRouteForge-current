from __future__ import annotations

from typing import Iterable, List

from .geometry import cycle_distance, euclidean
from .models import Customer, Point, RoutePlan


def nearest_neighbor(depot: Point, customers: Iterable[Customer]) -> List[Customer]:
    remaining = list(customers)
    if not remaining:
        return []

    ordered: list[Customer] = []
    current = depot
    while remaining:
        nxt = min(remaining, key=lambda c: euclidean(current, c.point))
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt.point
    return ordered


def two_opt_pass(depot: Point, route: List[Customer]) -> List[Customer]:
    best = route[:]
    best_cost = cycle_distance([c.point for c in best], depot=depot)
    n = len(best)

    for i in range(0, n - 2):
        for j in range(i + 2, n):
            candidate = best[:]
            candidate[i : j + 1] = reversed(candidate[i : j + 1])
            cand_cost = cycle_distance([c.point for c in candidate], depot=depot)
            if cand_cost + 1e-9 < best_cost:
                best = candidate
                best_cost = cand_cost
    return best


def solve_tsp_ortools(
    depot: Point,
    customers: Iterable[Customer],
    time_limit_sec: int = 2,
) -> List[Customer]:
    """Solve one vehicle's closed TSP route with OR-Tools.

    OR-Tools is optional at import time so the legacy heuristic remains usable
    in lightweight environments.
    """
    customers = list(customers)
    if len(customers) <= 2:
        return nearest_neighbor(depot, customers)

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError(
            "OR-Tools is not installed. Install the 'ortools' package or use "
            "routing_method='heuristic'."
        ) from exc

    points = [depot] + [customer.point for customer in customers]
    scale = 1_000_000
    matrix = [
        [
            int(round(euclidean(left, right) * scale))
            for right in points
        ]
        for left in points
    ]
    manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = max(1, int(time_limit_sec))
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools could not construct a TSP route.")

    ordered: list[Customer] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != 0:
            ordered.append(customers[node - 1])
        index = solution.Value(routing.NextVar(index))
    return ordered


def build_route_plans(
    assignments: dict[int, list[Customer]],
    depot: Point,
    two_opt_rounds: int = 2,
    routing_method: str = "heuristic",
    ortools_time_limit_sec: int = 2,
    ortools_fallback: bool = True,
) -> list[RoutePlan]:
    plans: list[RoutePlan] = []
    routing_method = (routing_method or "heuristic").lower().strip()
    if routing_method not in {"heuristic", "ortools"}:
        raise ValueError("routing_method must be 'heuristic' or 'ortools'.")

    for vehicle_id, customers in sorted(assignments.items(), key=lambda kv: kv[0]):
        if routing_method == "ortools":
            try:
                ordered = solve_tsp_ortools(
                    depot=depot,
                    customers=customers,
                    time_limit_sec=ortools_time_limit_sec,
                )
            except RuntimeError:
                if not ortools_fallback:
                    raise
                ordered = nearest_neighbor(depot, customers)
                for _ in range(max(0, two_opt_rounds)):
                    ordered = two_opt_pass(depot, ordered)
        else:
            ordered = nearest_neighbor(depot, customers)
            for _ in range(max(0, two_opt_rounds)):
                ordered = two_opt_pass(depot, ordered)

        plans.append(
            RoutePlan(
                vehicle_id=vehicle_id + 1,
                customers=ordered,
                load=sum(c.demand for c in ordered),
                distance=cycle_distance([c.point for c in ordered], depot=depot),
            )
        )
    return plans
