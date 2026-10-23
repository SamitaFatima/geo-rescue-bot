"""
risk_astar.py - RescueBot 2.0 (Member 3: Routing & Backend)

Risk-aware A* path search for the flood scenario.

WHY THIS EXISTS
---------------
The legacy searches in search_algorithms.py (BFS / DFS / Bidirectional)
only know "blocked" or "not blocked": they happily route the robot
through cells the flood model rates LOW or MEDIUM risk because those
cells are not blocked yet. This module adds a search that prefers
*safer* cells and detours around rising water when a detour is cheap
enough, while still using the same grid, the same blocked-cell rules
and the same return shape as the old algorithms.

HOW IT PLUGS IN (nothing existing is changed)
----------------------------------------------
Every legacy search has the signature

    search_fn(env, start, goal, valid_cells=None, explored_log=None)
        -> (path_or_None, nodes_explored, seconds)

`make_risk_astar(risk_grid)` returns a function with exactly that
signature, so it works unchanged with RescueAgent.plan(), with
csp.plan_with_csp() and with algorithm_comparison. `risk_astar()` is the
same search with the risk grid passed explicitly and additionally
returns the path cost.

RISK SCALE (shared with Member 2's hazard_engine.py - do not change)
--------------------------------------------------------------------
    0 safe   1 low   2 medium   3 high / blocked

COST MODEL
----------
Entering a cell costs 1 (one move) plus a risk penalty:

    risk 0 -> +0     risk 1 -> +1     risk 2 -> +4     risk 3 -> impassable

Because every move costs at least 1, the Manhattan-distance heuristic
never over-estimates, so A* returns the path with the LOWEST TOTAL COST
(optimal for this cost model). With an all-zero risk grid the cost is
simply the move count and the result is a shortest path, exactly like BFS.

Ties are broken deterministically (cost, then heuristic, then insertion
order) so the same inputs always produce the same route - required for a
reproducible demo and for tests.
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable, Sequence
from typing import Any

RISK_SAFE, RISK_LOW, RISK_MEDIUM, RISK_HIGH = 0, 1, 2, 3

# Extra cost, on top of the 1-per-move base cost, for ENTERING a cell of
# the given risk level. RISK_HIGH is absent on purpose: it is impassable.
RISK_PENALTY: dict[int, int] = {RISK_SAFE: 0, RISK_LOW: 1, RISK_MEDIUM: 4}

RiskGrid = Sequence[Sequence[int]]
Cell = tuple[int, int]


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _risk_at(risk_grid: RiskGrid | None, cell: Cell) -> int:
    if risk_grid is None:
        return RISK_SAFE
    return int(risk_grid[cell[0]][cell[1]])


def step_cost(risk_grid: RiskGrid | None, cell: Cell) -> int | None:
    """Cost of entering `cell`, or None if the flood makes it impassable."""
    risk = _risk_at(risk_grid, cell)
    if risk >= RISK_HIGH:
        return None
    return 1 + RISK_PENALTY.get(risk, 0)


def path_cost(path: Sequence[Cell] | None, risk_grid: RiskGrid | None) -> int | None:
    """Total cost of a path under the cost model above (the start cell is
    free; every later cell is charged on entry). Returns None for a missing
    path or a path that crosses an impassable (risk 3) cell - used to score
    legacy BFS routes on the same scale as risk-aware A*."""
    if not path:
        return None
    total = 0
    for cell in path[1:]:
        cost = step_cost(risk_grid, cell)
        if cost is None:
            return None
        total += cost
    return total


def path_risk_summary(path: Sequence[Cell] | None, risk_grid: RiskGrid | None) -> dict[str, int]:
    """How many cells (after the start) of each risk level the path enters."""
    summary = {"safe_cells": 0, "low_risk_cells": 0, "medium_risk_cells": 0,
               "high_risk_cells": 0}
    if not path:
        return summary
    names = {RISK_SAFE: "safe_cells", RISK_LOW: "low_risk_cells",
             RISK_MEDIUM: "medium_risk_cells", RISK_HIGH: "high_risk_cells"}
    for cell in path[1:]:
        summary[names[min(_risk_at(risk_grid, cell), RISK_HIGH)]] += 1
    return summary


def risk_astar(
    env: Any,
    start: Cell,
    goal: Cell,
    risk_grid: RiskGrid | None = None,
    valid_cells: set[Cell] | None = None,
    explored_log: list[Cell] | None = None,
) -> tuple[list[Cell] | None, int, float, int | None]:
    """A* over the environment grid with flood-risk-weighted step costs.

    Returns (path, nodes_explored, seconds, total_cost).
    `path` runs start..goal inclusive, or is None when no route exists
    (then total_cost is None). A cell is enterable when
      * it is inside the grid,
      * env.is_blocked() is False (walls and blocked/flooded hazard cells),
      * the risk grid does not rate it 3 (high / blocked),
      * it is in `valid_cells` when that set is given (CSP safety domain).
    The goal is always enterable if it is not a wall/hazard, matching how
    the legacy searches treat a victim or exit cell.
    """
    t0 = time.perf_counter()

    def done(path, explored, cost):
        return path, explored, time.perf_counter() - t0, cost

    if not env.in_bounds(start) or not env.in_bounds(goal):
        return done(None, 0, None)
    if start == goal:
        if explored_log is not None:
            explored_log.append(start)
        return done([start], 1, 0)

    counter = itertools.count()          # insertion order = deterministic ties
    open_heap: list[tuple[int, int, int, Cell]] = []
    h0 = _manhattan(start, goal)
    heapq.heappush(open_heap, (h0, h0, next(counter), start))
    best_cost: dict[Cell, int] = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    closed: set[Cell] = set()
    explored = 0

    while open_heap:
        _, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        explored += 1
        if explored_log is not None:
            explored_log.append(cur)

        if cur == goal:
            path: list[Cell] = []
            node: Cell | None = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return done(path, explored, best_cost[cur])

        for nxt in env.get_neighbors(cur):
            if nxt in closed or env.is_blocked(nxt):
                continue
            if valid_cells is not None and nxt not in valid_cells:
                continue
            cost = step_cost(risk_grid, nxt)
            if cost is None:
                continue
            new_cost = best_cost[cur] + cost
            if new_cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_cost
                parent[nxt] = cur
                h = _manhattan(nxt, goal)
                heapq.heappush(open_heap, (new_cost + h, h, next(counter), nxt))

    return done(None, explored, None)


def make_risk_astar(
    risk_grid: RiskGrid | None,
) -> Callable[..., tuple[list[Cell] | None, int, float]]:
    """Return a search function with the SAME signature and return shape as
    search_algorithms.bfs/dfs/bidirectional_search, bound to `risk_grid`:

        search_fn(env, start, goal, valid_cells=None, explored_log=None)
            -> (path_or_None, nodes_explored, seconds)

    The grid is captured by reference, so a caller that keeps one search
    function and swaps in a new snapshot should simply build a new one with
    this factory (cheap) after each hazard update.
    """

    def search(env, start, goal, valid_cells=None, explored_log=None):
        path, explored, seconds, _ = risk_astar(
            env, start, goal, risk_grid=risk_grid,
            valid_cells=valid_cells, explored_log=explored_log)
        return path, explored, seconds

    search.__name__ = "risk_aware_astar"
    return search
