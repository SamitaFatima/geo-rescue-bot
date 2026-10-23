"""
simulation_service.py - RescueBot 2.0 (Member 3: Routing & Backend)

WHAT THIS FILE DOES
--------------------
Runs the time-stepped rescue simulation the dashboard and the AI
assistant talk to. It glues together, WITHOUT modifying them:

  * Member 2's flood snapshots (gis_service.GISService, hour 0/2/5/10)
  * the existing Environment / RescueAgent / RescueCSP (rescue logic)
  * the new risk-aware A* (risk_astar.py) for route planning
  * the legacy BFS, run side-by-side purely as a comparison baseline

Each scenario has one Simulation. The cycle is:

    start   -> load hour-0 flood, plan the first route
    step    -> advance to the next snapshot (0 -> 2 -> 5 -> 10 h), apply the
               new flood to the grid, RE-PLAN, then let the robot move for the
               simulated time that passed (ROBOT_MOVES_PER_HOUR cells/hour)
    replan  -> force a re-plan against the current snapshot on request
    state   -> JSON snapshot of agents / victims / hazards / routes / metrics

Everything is deterministic: same scenario -> same layout, same snapshots,
same routes. All returned values are plain JSON (lists, dicts, numbers,
strings, booleans, null); cells are [row, col] in the SAME convention as
environment.py and Member 2's grids (row 0 = top, col 0 = left).

THREE CLASSES
-------------
  Simulation          one scenario's running state (not thread-safe by itself)
  SimulationService   thread-safe registry of simulations; used by api.py
  RescueServiceAdapter
                      the exact object Member 1's ai_tools.RescueBackendAdapter
                      protocol expects (4 methods, same names/arguments)

All outputs include the label "Illustrative scenario estimates, not verified
forecasts." as the team report requires.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import geo_data
from agent import DONE, FAILED, RescueAgent
from csp import RescueCSP, plan_with_csp
from environment import EMPTY, WALL, Environment
from gis_service import GISService
from risk_astar import make_risk_astar, path_cost, path_risk_summary
from search_algorithms import bfs

DISCLAIMER = "Illustrative scenario estimates, not verified forecasts."

# Simulated robot speed. Between two snapshots the robot moves
# (hours elapsed * ROBOT_MOVES_PER_HOUR) cells.
ROBOT_MOVES_PER_HOUR = 2
# Moves made by a step taken after the last snapshot (flood no longer changes).
OVERTIME_MOVES = 10
ROBOT_MAX_ENERGY = 100

# ----------------------------------------------------------------------
# The prepared rescue layout (cells are (row, col); the river is column 2
# and is always blocked, so the whole mission is east of it).
# Walls are "buildings"; victims are the people to rescue.
# ----------------------------------------------------------------------
LAYOUT: dict[str, Any] = {
    "start": (9, 9),
    "exit": (9, 5),
    "walls": [(0, 5), (0, 3), (1, 7)],
    # (cell, priority) in the order they are registered. The victims sit
    # close to the river, so the flood reaches the cells around them first.
    "victims": [((5, 3), "critical"), ((2, 3), "normal"), ((5, 4), "normal")],
}


class SimulationError(Exception):
    """Domain error with a stable machine-readable `code`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _cell(pos) -> list[int]:
    return [int(pos[0]), int(pos[1])]


def _cells(path) -> list[list[int]]:
    return [_cell(p) for p in path]


def _latlon(pos) -> dict[str, float]:
    return geo_data.cell_to_latlon(int(pos[0]), int(pos[1]))


def build_environment(layout: Mapping[str, Any] = LAYOUT, rows: int = geo_data.GRID_HEIGHT,
                      cols: int = geo_data.GRID_WIDTH) -> Environment:
    """Build the deterministic flood-rescue Environment (no random cells)."""
    env = Environment("Flood", rows=rows, cols=cols, num_victims=0, seed=0)
    env.grid = [[EMPTY] * cols for _ in range(rows)]
    env.agent_start = tuple(layout["start"])
    env.exit_pos = tuple(layout["exit"])
    env.grid[env.exit_pos[0]][env.exit_pos[1]] = 4  # EXIT
    for r, c in layout["walls"]:
        env.grid[r][c] = WALL
    env.victims = []
    env.priorities = {}
    for pos, priority in layout["victims"]:
        pos = tuple(pos)
        env.victims.append(pos)
        env.priorities[pos] = priority
        env.grid[pos[0]][pos[1]] = 3  # VICTIM
    env.num_victims = len(env.victims)
    return env


class Simulation:
    """One scenario's rescue run. Methods return JSON-ready dicts."""

    def __init__(self, scenario_id: str, gis: GISService | None = None,
                 layout: Mapping[str, Any] | None = None) -> None:
        self.scenario_id = geo_data.get_scenario(scenario_id).scenario_id  # raises if unknown
        self._gis = gis or GISService()
        bundle = self._gis.get_hazard_snapshot(self.scenario_id)
        self.assumptions: list[str] = list(bundle["assumptions"])
        self.snapshots: list[dict[str, Any]] = bundle["snapshots"]
        if not self.snapshots:
            raise SimulationError("NO_SNAPSHOTS", "The hazard engine returned no snapshots.")

        self.env = build_environment(layout or LAYOUT)
        self.agent = RescueAgent(self.env, max_energy=ROBOT_MAX_ENERGY)
        self.csp = RescueCSP(self.env)
        self.all_victims = [(v, self.env.priorities.get(v, "normal")) for v in self.env.victims]
        self.snapshot_index = 0
        self.overtime_steps = 0
        self.events: list[dict[str, Any]] = []
        self.route_changes: list[dict[str, Any]] = []
        self.risk_exposure = 0            # sum of risk of every cell the robot entered
        self.victim_order: list[list[int]] = []
        self.last_comparison: dict[str, Any] | None = None
        self.last_cost: int | None = None
        self.target: tuple[int, int] | None = None

        self._load_snapshot(0)
        self._plan(reason="Initial route planned at hour 0.", trigger="start")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> dict[str, Any]:
        return self.snapshots[self.snapshot_index]

    @property
    def hours(self) -> int:
        return int(self.snapshot["hours"])

    @property
    def finished(self) -> bool:
        return self.agent.state in (DONE, FAILED)

    @property
    def has_next_snapshot(self) -> bool:
        return self.snapshot_index + 1 < len(self.snapshots)

    def _load_snapshot(self, index: int) -> list[tuple[int, int]]:
        self.snapshot_index = index
        return self.env.apply_hazard_snapshot(
            self.snapshot["risk_grid"], protected={self.agent.position})

    def _event(self, kind: str, message: str, **extra: Any) -> dict[str, Any]:
        event = {"event_id": len(self.events) + 1, "type": kind, "hours": self.hours,
                 "step_index": self.snapshot_index, "message": message}
        event.update(extra)
        self.events.append(event)
        return event

    def _current_route(self) -> list[tuple[int, int]]:
        return [self.agent.position] + list(self.agent.path)

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def _plan(self, reason: str, trigger: str,
              newly_blocked: list[tuple[int, int]] | None = None) -> dict[str, Any]:
        """(Re)plan with CSP + risk-aware A*; record a route-change event
        when the route differs from the previous one."""
        risk_grid = self.snapshot["risk_grid"]
        old_route = self._current_route() if self.agent.path else []
        old_target = self.target
        old_cost = path_cost(old_route, risk_grid) if old_route else None
        was_blocked = bool(old_route) and any(
            self.env.is_blocked(p) for p in old_route[1:])

        ok = plan_with_csp(self.agent, make_risk_astar(risk_grid), self.csp)
        applied = self.csp.apply(self.agent)
        self.victim_order = _cells(applied["victim_order"]) if applied["ok"] else []

        if not ok:
            self.target = None
            message = f"No safe route could be planned: {self.agent.fail_reason}."
            event = self._event("route_failed", message, trigger=trigger)
            return {"changed": True, "ok": False, "reason": message, "event": event}

        self.target = self.agent.current_goal()
        new_route = self._current_route()
        new_cost = path_cost(new_route, risk_grid)
        self.last_cost = new_cost
        self.last_comparison = self._compare_with_bfs(new_route, risk_grid)

        changed = (not old_route) or old_route != new_route
        blocked_on_old = [p for p in old_route[1:] if p in (newly_blocked or [])]
        blocking_cells = blocked_on_old or (
            [p for p in old_route[1:] if self.env.is_blocked(p)] if was_blocked else [])
        if not old_route:
            detail = reason
        elif was_blocked or blocked_on_old:
            detail = (f"Previous route to {_cell(old_target) if old_target else 'target'} "
                      f"crossed flooded cell(s) {_cells(blocking_cells)} (now blocked at "
                      f"{self.hours} h). Replanned a safer route to "
                      f"{_cell(self.target)} (new route cost {new_cost}"
                      + (f", was {old_cost}" if old_cost is not None else
                         ", the old route is no longer passable") + ").")
        elif changed:
            old_risk = path_risk_summary(old_route, risk_grid)
            new_risk = path_risk_summary(new_route, risk_grid)
            detail = (f"Flood risk rose along the previous route; switched to a "
                      f"lower-risk route to {_cell(self.target)}. Medium-risk "
                      f"cells entered {old_risk['medium_risk_cells']} -> "
                      f"{new_risk['medium_risk_cells']}, cost {old_cost} -> {new_cost}.")
        else:
            detail = (f"Route to {_cell(self.target)} re-checked at {self.hours} h "
                      f"and is still the lowest-risk option (cost {new_cost}).")

        event = self._event(
            "route_planned" if not old_route else ("route_changed" if changed else "route_confirmed"),
            detail, trigger=trigger, reason=reason, changed=changed,
            previous_cost=old_cost, new_cost=new_cost,
            previous_route=_cells(old_route), new_route=_cells(new_route),
            target=_cell(self.target), newly_blocked_cells=_cells(newly_blocked or []),
            blocking_cells=_cells(blocking_cells))
        if changed and old_route:
            self.route_changes.append(event)
        return {"changed": changed, "ok": True, "reason": detail, "event": event}

    def _compare_with_bfs(self, route, risk_grid) -> dict[str, Any]:
        """Score the chosen risk-aware route against what legacy BFS would do
        for the same leg, on the same cost scale (kept for the report/demo)."""
        goal = route[-1]
        valid = self.csp.get_valid_cells()
        bfs_path, bfs_explored, _ = bfs(self.env, self.agent.position, goal, valid_cells=valid)
        return {
            "goal": _cell(goal),
            "risk_astar": {"cost": path_cost(route, risk_grid), "moves": len(route) - 1,
                           **path_risk_summary(route, risk_grid)},
            "bfs": {"cost": path_cost(bfs_path, risk_grid),
                    "moves": (len(bfs_path) - 1) if bfs_path else None,
                    "nodes_explored": bfs_explored,
                    **path_risk_summary(bfs_path, risk_grid)},
        }

    # ------------------------------------------------------------------
    # advancing
    # ------------------------------------------------------------------
    def _move(self, max_moves: int) -> int:
        moved = 0
        guard = max_moves * 4 + 10
        while moved < max_moves and not self.finished and guard > 0:
            guard -= 1
            status = self.agent.step()
            if status in ("NEED_PLAN", "REPLAN"):
                self._plan(reason="Robot needed a new route.", trigger="agent_replan")
                continue
            if status in ("MOVED", "VICTIM_RESCUED", DONE):
                moved += 1
                pos = self.agent.position
                self.risk_exposure += int(self.env.risk_grid[pos[0]][pos[1]])
                if status == "VICTIM_RESCUED":
                    self._event("victim_rescued", f"Victim rescued at {_cell(pos)}.",
                                position=_cell(pos))
                    self._plan(reason="Next target after rescue.", trigger="after_rescue")
                elif status == DONE:
                    self._event("mission_done", "All victims rescued and robot reached the exit.")
            elif status == FAILED:
                self._event("mission_failed", f"Mission failed: {self.agent.fail_reason}.")
                break
        return moved

    def step(self) -> dict[str, Any]:
        """Advance to the next snapshot (or continue, after the last one)."""
        if self.finished:
            return {"advanced": False, "moves": 0,
                    "message": f"Simulation already finished ({self.agent.state}).",
                    "newly_blocked_cells": []}
        newly_blocked: list[tuple[int, int]] = []
        if self.has_next_snapshot:
            prev_hours = self.hours
            newly_blocked = self._load_snapshot(self.snapshot_index + 1)
            elapsed = self.hours - prev_hours
            self._event("hazard_update",
                        f"Flood snapshot advanced to {self.hours} h: "
                        f"{len(newly_blocked)} newly blocked cell(s).",
                        newly_blocked_cells=_cells(newly_blocked))
            self._plan(reason=f"Flood snapshot updated to {self.hours} h.",
                       trigger="hazard_update", newly_blocked=newly_blocked)
            moves = self._move(elapsed * ROBOT_MOVES_PER_HOUR)
        else:
            self.overtime_steps += 1
            moves = self._move(OVERTIME_MOVES)
        return {"advanced": True, "moves": moves, "newly_blocked_cells": _cells(newly_blocked)}

    def replan(self, reason: str | None = None) -> dict[str, Any]:
        if self.finished:
            raise SimulationError("SIMULATION_FINISHED",
                                  f"Simulation already finished ({self.agent.state}).")
        why = (reason or "Manual re-plan requested.").strip()[:200]
        result = self._plan(reason=why, trigger="manual_replan")
        return result

    # ------------------------------------------------------------------
    # JSON views
    # ------------------------------------------------------------------
    def route_view(self) -> dict[str, Any]:
        route = self._current_route() if self.agent.path else []
        risk_grid = self.snapshot["risk_grid"]
        if self.agent.state == FAILED:
            status = "FAILED"
        elif self.agent.state == DONE:
            status = "COMPLETE"
        elif not route:
            status = "NONE"
        elif any(self.env.is_blocked(p) for p in route[1:]):
            status = "BLOCKED"
        else:
            status = "OPEN"
        return {
            "route_id": "route-1",
            "status": status,
            "algorithm": "risk_aware_astar",
            "target": _cell(self.target) if self.target else None,
            "target_type": ("victim" if self.env.victims else "exit") if self.target else None,
            "path": _cells(route),
            "path_latlon": [_latlon(p) for p in route],
            "cost": path_cost(route, risk_grid) if route else None,
            "risk_cells": path_risk_summary(route, risk_grid) if route else None,
            "victim_order": self.victim_order,
        }

    def state(self) -> dict[str, Any]:
        a = self.agent
        snap = self.snapshot
        return {
            "scenario_id": self.scenario_id,
            "status": a.state if self.finished else ("RUNNING" if self.snapshot_index or a.steps_taken
                                                      else "READY"),
            "hours": self.hours,
            "step_index": self.snapshot_index,
            "total_steps": len(self.snapshots),
            "has_next_step": self.has_next_snapshot,
            "agents": [{
                "agent_id": "robot-1",
                "position": _cell(a.position),
                "latlon": _latlon(a.position),
                "state": a.state,
                "energy": a.energy,
                "carrying_victims": a.carrying_victims,
                "trail": _cells(a.trail),
                "fail_reason": a.fail_reason or None,
            }],
            "start": _cell(self.env.agent_start),
            "exit": {"position": _cell(self.env.exit_pos), "latlon": _latlon(self.env.exit_pos)},
            "victims": [{
                "position": _cell(v), "latlon": _latlon(v), "priority": priority,
                "rescued": v not in self.env.victims,
                "risk": int(snap["risk_grid"][v[0]][v[1]]),
            } for v, priority in self.all_victims],
            "walls": [_cell((r, c)) for r in range(self.env.rows)
                      for c in range(self.env.cols) if self.env.grid[r][c] == WALL],
            "hazards": {
                "hours": self.hours,
                "risk_grid": snap["risk_grid"],
                "blocked_cells": snap["blocked_cells"],
                "exposure": snap["exposure"],
                "source_timestamp": snap["source_timestamp"],
                "confidence": snap["confidence"],
            },
            "routes": [self.route_view()],
            "route_changes": [{"event_id": e["event_id"], "hours": e["hours"],
                               "reason": e["message"]} for e in self.route_changes],
            "events": self.events[-20:],
            "metrics": {
                "steps_taken": a.steps_taken,
                "victims_rescued": a.victims_rescued,
                "victims_total": a.total_victims_target,
                "victims_remaining": len(self.env.victims),
                "route_changes": len(self.route_changes),
                "robot_blocked_replans": a.replan_count,
                "risk_exposure_along_trail": self.risk_exposure,
                "search_nodes_explored": a.total_nodes_explored,
                "last_route_cost": self.last_cost,
                "last_comparison": self.last_comparison,
            },
            "assumptions": self.assumptions,
            "disclaimer": DISCLAIMER,
        }

    def explanation(self) -> dict[str, Any]:
        change = self.route_changes[-1] if self.route_changes else None
        return {
            "scenario_id": self.scenario_id,
            "route_id": "route-1",
            "route_changed": change is not None,
            "reason": change["message"] if change else
            "No route change has happened yet; the current route is still the "
            "lowest-risk option for the current flood snapshot.",
            "hours": change["hours"] if change else self.hours,
            "previous_route": change["previous_route"] if change else None,
            "new_route": change["new_route"] if change else _cells(self._current_route()),
            "previous_cost": change["previous_cost"] if change else None,
            "new_cost": change["new_cost"] if change else self.last_cost,
            "blocking_cells": change["blocking_cells"] if change else [],
            "newly_blocked_cells": change["newly_blocked_cells"] if change else [],
            "total_route_changes": len(self.route_changes),
            "disclaimer": DISCLAIMER,
        }


class SimulationService:
    """Thread-safe registry of simulations, one per scenario_id."""

    def __init__(self, gis: GISService | None = None) -> None:
        self._gis = gis or GISService()
        self._sims: dict[str, Simulation] = {}
        self._lock = threading.RLock()

    def _canonical(self, scenario_id: str) -> str:
        try:
            return geo_data.get_scenario(scenario_id).scenario_id
        except geo_data.UnknownScenarioError as exc:
            raise SimulationError("UNKNOWN_SCENARIO", str(exc)) from exc

    def _get(self, scenario_id: str) -> Simulation:
        sid = self._canonical(scenario_id)
        sim = self._sims.get(sid)
        if sim is None:
            raise SimulationError("NOT_STARTED",
                                  f"No simulation is running for '{sid}'. Start one first.")
        return sim

    def is_started(self, scenario_id: str) -> bool:
        with self._lock:
            return self._canonical(scenario_id) in self._sims

    def start(self, scenario_id: str, restart: bool = False) -> dict[str, Any]:
        """Idempotent: starting an already-running scenario returns the
        existing state (`already_started: true`) instead of resetting it,
        so a retried HTTP call or duplicate delivery cannot undo progress.
        Pass restart=True to deliberately reset."""
        with self._lock:
            sid = self._canonical(scenario_id)
            already = sid in self._sims and not restart
            if not already:
                self._sims[sid] = Simulation(sid, self._gis)
            sim = self._sims[sid]
            return {"scenario_id": sid, "started": True, "already_started": already,
                    "simulation_id": f"sim-{sid}", "initial_state": sim.state(),
                    "disclaimer": DISCLAIMER}

    def state(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get(scenario_id).state()

    def step(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            sim = self._get(scenario_id)
            outcome = sim.step()
            snap = sim.snapshot
            return {"scenario_id": sim.scenario_id, **outcome,
                    "snapshot": {"hours": snap["hours"], "risk_grid": snap["risk_grid"],
                                 "blocked_cells": snap["blocked_cells"],
                                 "exposure": snap["exposure"]},
                    "state": sim.state()}

    def replan(self, scenario_id: str, reason: str | None = None) -> dict[str, Any]:
        with self._lock:
            sim = self._get(scenario_id)
            result = sim.replan(reason)
            route = sim.route_view()
            return {"scenario_id": sim.scenario_id, "ok": result["ok"],
                    "changed": result["changed"], "route": route, "cost": route["cost"],
                    "reason": result["reason"], "disclaimer": DISCLAIMER}

    def explain(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get(scenario_id).explanation()


class RescueServiceAdapter:
    """Implements ai_tools.RescueBackendAdapter (Member 1's protocol) with
    the exact method names and parameters. Construct with no arguments, or
    pass a shared SimulationService so the API and the assistant see the
    SAME simulation:

        service = SimulationService()
        rescue = RescueServiceAdapter(service)

    Read-only methods never start a simulation. Problems raise an exception;
    Member 1's ToolRegistry turns it into a sanitized SERVICE_UNAVAILABLE."""

    def __init__(self, service: SimulationService | None = None) -> None:
        self._service = service or SimulationService()

    def start_rescue_simulation(self, scenario_id: str) -> Mapping[str, Any]:
        result = self._service.start(scenario_id)      # idempotent
        state = result["initial_state"]
        return {"scenario_id": result["scenario_id"], "started": True,
                "already_started": result["already_started"],
                "simulation_id": result["simulation_id"], "state": "RUNNING",
                "route": state["routes"][0], "disclaimer": DISCLAIMER}

    def get_rescue_routes(self, scenario_id: str) -> Mapping[str, Any]:
        if not self._service.is_started(scenario_id):
            return {"scenario_id": scenario_id, "started": False, "routes": [],
                    "detail": "No simulation has been started for this scenario."}
        state = self._service.state(scenario_id)
        return {"scenario_id": state["scenario_id"], "started": True,
                "hours": state["hours"], "routes": state["routes"],
                "disclaimer": DISCLAIMER}

    def get_simulation_status(self, scenario_id: str) -> Mapping[str, Any]:
        if not self._service.is_started(scenario_id):
            return {"scenario_id": scenario_id, "status": "NOT_STARTED",
                    "victims_rescued": 0,
                    "detail": "No simulation has been started for this scenario."}
        state = self._service.state(scenario_id)
        m = state["metrics"]
        return {"scenario_id": state["scenario_id"], "status": state["status"],
                "hours": state["hours"], "victims_rescued": m["victims_rescued"],
                "victims_total": m["victims_total"],
                "victims_remaining": m["victims_remaining"],
                "route_changes": m["route_changes"],
                "robot_position": state["agents"][0]["position"],
                "energy": state["agents"][0]["energy"], "disclaimer": DISCLAIMER}

    def explain_route_change(self, scenario_id: str) -> Mapping[str, Any]:
        if not self._service.is_started(scenario_id):
            return {"scenario_id": scenario_id, "route_changed": False, "reason": None,
                    "detail": "No simulation has been started for this scenario."}
        return self._service.explain(scenario_id)
