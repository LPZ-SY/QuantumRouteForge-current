from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Customer:
    customer_id: int
    x: float
    y: float
    demand: int

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class DispatchInstance:
    depot: Point
    customers: List[Customer]
    num_vehicles: int
    vehicle_capacity: int

    @property
    def total_demand(self) -> int:
        return sum(c.demand for c in self.customers)

    @property
    def feasible_capacity(self) -> bool:
        return self.total_demand <= self.num_vehicles * self.vehicle_capacity


@dataclass(frozen=True)
class AssignmentMetadata:
    requested_mode: str
    used_mode: str
    energy: float
    message: str
    quantum_task_id: Optional[str] = None
    quantum_backend: Optional[str] = None
    quantum_bitstring: Optional[str] = None
    quantum_endpoint: Optional[str] = None
    quantum_task_ids: Optional[List[str]] = None
    clustering_method: Optional[str] = None
    routing_method: Optional[str] = None
    qaoa_subproblem_policy: Optional[str] = None
    qaoa_subproblem_size: Optional[int] = None
    tabu_iterations: int = 0
    tabu_accepted_moves: int = 0
    initial_proxy_distance: Optional[float] = None
    best_proxy_distance: Optional[float] = None
    quantum_counts: Optional[Dict[str, int]] = None
    quantum_iteration_records: Optional[List[Dict[str, object]]] = None
    stratified_ablation: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class RoutePlan:
    vehicle_id: int
    customers: List[Customer]
    load: int
    distance: float


@dataclass(frozen=True)
class OptimizationResult:
    instance: DispatchInstance
    assignments: Dict[int, List[Customer]]
    routes: List[RoutePlan]
    total_distance: float
    metadata: AssignmentMetadata
