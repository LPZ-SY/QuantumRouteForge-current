from __future__ import annotations

import math
from typing import Iterable, Tuple

Point = Tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cycle_distance(points: Iterable[Point], depot: Point) -> float:
    points = list(points)
    if not points:
        return 0.0

    total = euclidean(depot, points[0])
    for i in range(len(points) - 1):
        total += euclidean(points[i], points[i + 1])
    total += euclidean(points[-1], depot)
    return total
