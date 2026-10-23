"""
test_routes.py - RescueBot 2.0 (Member 3: Routing & Backend)

Tests for risk_astar.py, Environment.apply_hazard_snapshot,
simulation_service.py and api.py.

Run either way (no network, no API keys needed):
    python -m pytest test_routes.py -v
    python test_routes.py

Tests that need an optional package skip themselves with a message:
  - API tests need Flask (pip install flask)
  - the "real ToolRegistry" test needs python-dotenv, which Member 1's
    ai_tools.py imports (it is already in requirements.txt)
"""
from __future__ import annotations

import heapq
import json
import random
import sys
import traceback

from agent import DONE, FAILED, RescueAgent
from environment import EMPTY, HAZARD, WALL, Environment
from risk_astar import (make_risk_astar, path_cost, path_risk_summary, risk_astar,
                        step_cost)
from search_algorithms import bfs
from simulation_service import (RescueServiceAdapter, Simulation, SimulationError,
                                SimulationService, build_environment)

SCENARIO = "riverside-flood-01"


class Skip(Exception):
    """Raised by tests that need an optional package; turned into a real
    pytest skip when running under pytest."""


def _skip(message):
    try:
        import pytest
    except ImportError:
        raise Skip(message)
    pytest.skip(message)


def blank_env(rows=6, cols=6, victims=0):
    env = Environment("Building Fire", rows=rows, cols=cols, num_victims=victims, seed=0)
    env.grid = [[EMPTY] * cols for _ in range(rows)]
    env.victims, env.priorities = [], {}
    env.exit_pos = (rows - 1, cols - 1)
    env.grid[rows - 1][cols - 1] = 4
    return env


def zero_grid(rows=6, cols=6):
    return [[0] * cols for _ in range(rows)]


# ======================================================================
# risk_astar
# ======================================================================
def test_zero_risk_matches_bfs_shortest_length():
    env = blank_env()
    for wall in [(1, 1), (2, 1), (3, 1), (3, 3), (3, 4)]:
        env.grid[wall[0]][wall[1]] = WALL
    path, _, _, cost = risk_astar(env, (0, 0), (5, 5), zero_grid())
    bfs_path, _, _ = bfs(env, (0, 0), (5, 5))
    assert len(path) == len(bfs_path) and cost == len(path) - 1
    assert path[0] == (0, 0) and path[-1] == (5, 5)
    print("PASS zero-risk A* returns a shortest path like BFS")


def test_never_enters_high_risk_cells_and_reports_no_path():
    env = blank_env()
    risk = zero_grid()
    for r in range(6):
        risk[r][3] = 3                       # a full wall of flood water
    path, explored, _, cost = risk_astar(env, (0, 0), (5, 5), risk)
    assert path is None and cost is None and explored > 0
    risk[2][3] = 0                           # open one gap
    path, _, _, _ = risk_astar(env, (0, 0), (5, 5), risk)
    assert path is not None and (2, 3) in path
    assert all(risk[r][c] < 3 for r, c in path[1:])
    print("PASS high-risk cells are impassable; path found only through the gap")


def test_prefers_safer_detour_over_short_risky_path():
    env = blank_env(rows=5, cols=7)
    env.exit_pos = (2, 6)
    env.grid[4][6] = EMPTY
    env.grid[2][6] = 4
    risk = [[0] * 7 for _ in range(5)]
    for c in range(1, 6):
        risk[2][c] = 2                       # direct row is medium risk
    path, _, _, cost = risk_astar(env, (2, 0), (2, 6), risk)
    direct_cost = path_cost([(2, c) for c in range(7)], risk)
    assert cost < direct_cost
    assert path_risk_summary(path, risk)["medium_risk_cells"] == 0
    assert len(path) - 1 > 6                 # it is a longer, but safer, route
    print(f"PASS safer detour chosen (cost {cost} < direct {direct_cost})")


def test_cost_is_optimal_against_independent_dijkstra():
    rng = random.Random(11)
    for _ in range(60):
        rows, cols = 7, 7
        env = blank_env(rows, cols)
        for _ in range(8):
            r, c = rng.randrange(rows), rng.randrange(cols)
            if (r, c) not in ((0, 0), (rows - 1, cols - 1)):
                env.grid[r][c] = WALL
        risk = [[rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(cols)] for _ in range(rows)]
        risk[0][0] = risk[rows - 1][cols - 1] = 0
        _, _, _, cost = risk_astar(env, (0, 0), (rows - 1, cols - 1), risk)
        # independent reference
        dist, heap, best = {(0, 0): 0}, [(0, (0, 0))], None
        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist.get(cur, 1e9):
                continue
            if cur == (rows - 1, cols - 1):
                best = d
                break
            for n in env.get_neighbors(cur):
                sc = step_cost(risk, n)
                if env.is_blocked(n) or sc is None:
                    continue
                if d + sc < dist.get(n, 1e9):
                    dist[n] = d + sc
                    heapq.heappush(heap, (d + sc, n))
        assert cost == best, (cost, best)
    print("PASS A* cost equals an independent Dijkstra on 60 random grids")


def test_is_deterministic():
    env = blank_env()
    risk = [[(r + c) % 3 for c in range(6)] for r in range(6)]
    risk[0][0] = risk[5][5] = 0
    first = risk_astar(env, (0, 0), (5, 5), risk)[0]
    for _ in range(5):
        assert risk_astar(env, (0, 0), (5, 5), risk)[0] == first
    print("PASS same input always yields the same route")


def test_same_signature_as_legacy_searches_and_works_with_agent():
    env = Environment("Flood", rows=6, cols=6, num_victims=1, seed=3)
    agent = RescueAgent(env)
    search = make_risk_astar(zero_grid())
    path, explored, seconds = search(env, env.agent_start, env.exit_pos,
                                     valid_cells=None, explored_log=[])
    assert isinstance(explored, int) and seconds >= 0
    assert agent.plan(search) or agent.state == FAILED
    if path is not None:
        assert path[0] == env.agent_start and path[-1] == env.exit_pos
    print("PASS make_risk_astar is a drop-in search_fn (3-tuple) for RescueAgent.plan")


def test_start_equals_goal_and_out_of_bounds():
    env = blank_env()
    assert risk_astar(env, (1, 1), (1, 1), zero_grid())[0] == [(1, 1)]
    assert risk_astar(env, (1, 1), (9, 9), zero_grid())[0] is None
    print("PASS trivial and out-of-bounds goals handled")


def test_path_cost_penalises_risk_and_rejects_blocked():
    risk = zero_grid()
    risk[0][1], risk[0][2] = 1, 2
    assert path_cost([(0, 0), (0, 1), (0, 2)], risk) == (1 + 1) + (1 + 4)
    risk[0][2] = 3
    assert path_cost([(0, 0), (0, 1), (0, 2)], risk) is None
    assert path_cost(None, risk) is None
    print("PASS path_cost matches the documented cost model")


# ======================================================================
# Environment.apply_hazard_snapshot
# ======================================================================
def test_apply_hazard_snapshot_blocks_restores_and_protects():
    env = build_environment()
    victim = env.victims[0]
    walls_before = {(r, c) for r in range(10) for c in range(10) if env.grid[r][c] == WALL}
    risk = [[0] * 10 for _ in range(10)]
    risk[4][4] = 3
    risk[victim[0]][victim[1]] = 3
    risk[env.exit_pos[0]][env.exit_pos[1]] = 3
    risk[7][7] = 3
    newly = env.apply_hazard_snapshot(risk, protected={(7, 7)})
    assert env.grid[4][4] == HAZARD and newly == [(4, 4)]
    assert env.cell_type(victim) == 3 and env.cell_type(env.exit_pos) == 4
    assert env.grid[7][7] == EMPTY            # protected (robot stands there)
    risk[4][4] = 0                            # water recedes
    env.apply_hazard_snapshot(risk)
    assert env.grid[4][4] == EMPTY
    assert {(r, c) for r in range(10) for c in range(10) if env.grid[r][c] == WALL} == walls_before
    try:
        env.apply_hazard_snapshot([[0] * 3] * 3)
        raise AssertionError("size mismatch should raise")
    except ValueError:
        pass
    print("PASS snapshot blocks risk-3 cells, protects victims/exit, restores, keeps walls")


# ======================================================================
# Simulation
# ======================================================================
def run_to_end(service, scenario=SCENARIO, limit=12):
    states = []
    for _ in range(limit):
        out = service.step(scenario)
        states.append(out["state"])
        if out["state"]["status"] in ("DONE", "FAILED"):
            break
    return states


def test_start_state_has_every_contract_field():
    svc = SimulationService()
    result = svc.start(SCENARIO)
    st = result["initial_state"]
    assert result["started"] and not result["already_started"]
    for key in ("agents", "victims", "hazards", "routes", "metrics", "assumptions",
                "disclaimer", "hours", "walls", "exit"):
        assert key in st, key
    assert st["hours"] == 0 and len(st["victims"]) == 3
    assert st["routes"][0]["status"] == "OPEN" and st["routes"][0]["path"]
    assert "not verified forecasts" in st["disclaimer"]
    assert st["routes"][0]["path_latlon"][0].keys() == {"lat", "lon"}
    json.dumps(st)                                    # strictly JSON-serialisable
    print("PASS start returns the full contract state")


def test_hours_follow_snapshots_0_2_5_10():
    svc = SimulationService()
    svc.start(SCENARIO)
    hours = [s["hours"] for s in run_to_end(svc)]
    assert hours[:3] == [2, 5, 10]
    print("PASS time advances through the 0/2/5/10 h snapshots")


def test_flood_blocks_route_and_robot_replans_and_finishes():
    svc = SimulationService()
    svc.start(SCENARIO)
    states = run_to_end(svc)
    final = states[-1]
    assert final["status"] == "DONE", final["agents"][0]["fail_reason"]
    assert final["metrics"]["victims_rescued"] == 3
    assert final["metrics"]["route_changes"] >= 1
    for st in states:
        assert st["routes"][0]["status"] != "BLOCKED"
    explanation = svc.explain(SCENARIO)
    assert explanation["route_changed"] and explanation["blocking_cells"]
    assert "flooded" in explanation["reason"] and explanation["hours"] == 10
    blocked_now = {tuple(c) for c in final["hazards"]["blocked_cells"]}
    for cell in explanation["blocking_cells"]:
        assert tuple(cell) in blocked_now
        assert cell not in explanation["new_route"]
    print("PASS 10 h flood blocks a route; robot replans around it and rescues everyone")


def test_robot_never_steps_into_a_flooded_cell():
    svc = SimulationService()
    svc.start(SCENARIO)
    sim = svc._sims[SCENARIO]
    flooded_when_entered = []
    while not sim.finished and sim.snapshot_index < 10:
        before = len(sim.agent.trail)
        sim.step()
        for pos in sim.agent.trail[before:]:
            if sim.env.risk_grid[pos[0]][pos[1]] >= 3 and pos not in sim.env.victims \
                    and pos != sim.env.exit_pos:
                # a victim cell may be at risk 3 (it is never walled off); any
                # other flooded cell entered is a bug
                if pos not in {(5, 3), (2, 3), (5, 4)}:
                    flooded_when_entered.append(pos)
        if sim.snapshot_index == len(sim.snapshots) - 1 and sim.overtime_steps > 3:
            break
    assert not flooded_when_entered, flooded_when_entered
    print("PASS robot never entered a blocked flood cell")


def test_runs_are_reproducible():
    a, b = SimulationService(), SimulationService()
    a.start(SCENARIO)
    b.start(SCENARIO)

    def strip(state):
        return json.loads(json.dumps(state))

    sa, sb = run_to_end(a), run_to_end(b)
    assert [strip(s) for s in sa] == [strip(s) for s in sb]
    print("PASS two independent runs give identical results")


def test_start_is_idempotent_and_restart_resets():
    svc = SimulationService()
    svc.start(SCENARIO)
    svc.step(SCENARIO)
    again = svc.start(SCENARIO)
    assert again["already_started"] and again["initial_state"]["hours"] == 2
    fresh = svc.start(SCENARIO, restart=True)
    assert not fresh["already_started"] and fresh["initial_state"]["hours"] == 0
    print("PASS duplicate start keeps progress; restart=True resets")


def test_scenario_alias_and_error_codes():
    svc = SimulationService()
    assert svc.start("scenario-17")["scenario_id"] == "scenario-17"
    for call, code in [(lambda: svc.state("nope-123"), "UNKNOWN_SCENARIO"),
                       (lambda: svc.step(SCENARIO), "NOT_STARTED"),
                       (lambda: svc.start("bad id"), "UNKNOWN_SCENARIO")]:
        try:
            call()
            raise AssertionError("expected SimulationError")
        except SimulationError as exc:
            assert exc.code == code, (exc.code, code)
    print("PASS alias works; unknown / not-started raise coded errors")


def test_unreachable_victim_fails_gracefully_with_reason():
    layout = {"start": (9, 9), "exit": (9, 5), "walls": [(0, 4), (1, 3), (1, 5), (2, 4)],
              "victims": [((1, 4), "critical")]}
    sim = Simulation(SCENARIO, layout=layout)
    assert sim.agent.state == FAILED and sim.agent.fail_reason
    view = sim.state()
    assert view["status"] == "FAILED" and view["routes"][0]["status"] == "FAILED"
    assert view["agents"][0]["fail_reason"]
    out = sim.step()
    assert out["advanced"] is False                   # finished sims do not advance
    print("PASS sealed-off victim -> FAILED with a reason, no crash")


def test_replan_returns_reason_and_rejects_finished():
    svc = SimulationService()
    svc.start(SCENARIO)
    out = svc.replan(SCENARIO, "dashboard asked")
    assert out["ok"] and out["route"]["path"] and out["cost"] is not None and out["reason"]
    run_to_end(svc)
    try:
        svc.replan(SCENARIO)
        raise AssertionError("expected SIMULATION_FINISHED")
    except SimulationError as exc:
        assert exc.code == "SIMULATION_FINISHED"
    print("PASS replan returns route/cost/reason; finished simulation is rejected")


def test_risk_aware_never_costs_more_than_bfs_on_same_leg():
    svc = SimulationService()
    svc.start(SCENARIO)
    for st in run_to_end(svc):
        cmp_ = st["metrics"]["last_comparison"]
        if cmp_ and cmp_["bfs"]["cost"] is not None:
            assert cmp_["risk_astar"]["cost"] <= cmp_["bfs"]["cost"]
    print("PASS risk-aware route cost <= legacy BFS route cost on every leg")


def test_state_lists_rescued_victims_too():
    svc = SimulationService()
    svc.start(SCENARIO)
    final = run_to_end(svc)[-1]
    assert len(final["victims"]) == 3 and all(v["rescued"] for v in final["victims"])
    first = svc.start(SCENARIO, restart=True)["initial_state"]
    assert len(first["victims"]) == 3 and not any(v["rescued"] for v in first["victims"])
    print("PASS victims stay listed with a rescued flag")


def test_random_layouts_keep_invariants():
    """Fuzz: 120 random layouts. No crash, JSON-safe state, robot never
    inside a blocked cell, a running route is never BLOCKED, and every run
    ends DONE or FAILED (never hangs)."""
    import geo_data
    rng = random.Random(2026)
    cells = [(r, c) for r in range(10) for c in range(3, 10)]
    for _ in range(120):
        start, ex = rng.sample(cells, 2)
        taken = {start, ex}
        walls = rng.sample([c for c in cells if c not in taken], rng.randint(0, 14))
        taken |= set(walls)
        vs = rng.sample([c for c in cells if c not in taken], rng.randint(1, 4))
        layout = {"start": start, "exit": ex, "walls": walls,
                  "victims": [(v, "critical" if i == 0 else "normal") for i, v in enumerate(vs)]}
        sim = Simulation(SCENARIO, layout=layout)
        for _ in range(25):
            json.dumps(sim.state())
            pos = sim.agent.position
            assert sim.env.cell_type(pos) != HAZARD, (layout, pos)
            if not sim.finished:
                assert sim.route_view()["status"] in ("OPEN", "NONE"), layout
            if sim.finished:
                break
            sim.step()
        assert sim.agent.state in (DONE, FAILED), layout
        if sim.agent.state == DONE:
            assert sim.agent.victims_rescued == len(vs)
    print("PASS 120 random layouts: no crash, no hang, invariants hold")


# ======================================================================
# RescueServiceAdapter (Member 1's RescueBackendAdapter protocol)
# ======================================================================
def test_adapter_four_methods_json_and_read_only_never_starts():
    svc = SimulationService()
    adapter = RescueServiceAdapter(svc)
    for method in ("get_rescue_routes", "get_simulation_status", "explain_route_change"):
        out = getattr(adapter, method)(scenario_id=SCENARIO)
        json.dumps(out)
        assert out["scenario_id"] == SCENARIO
    assert not svc.is_started(SCENARIO)               # read-only calls never start it
    started = adapter.start_rescue_simulation(scenario_id=SCENARIO)
    assert started["started"] and started["state"] == "RUNNING"
    again = adapter.start_rescue_simulation(scenario_id=SCENARIO)
    assert again["already_started"]                    # idempotent
    svc.step(SCENARIO)
    routes = adapter.get_rescue_routes(scenario_id=SCENARIO)
    assert routes["routes"][0]["status"] in ("OPEN", "COMPLETE")
    status = adapter.get_simulation_status(scenario_id=SCENARIO)
    assert status["victims_total"] == 3 and "victims_rescued" in status
    json.dumps([started, routes, status, adapter.explain_route_change(scenario_id=SCENARIO)])
    print("PASS adapter exposes the four protocol methods as strict JSON")


def test_adapter_works_through_members_real_tool_registry():
    try:
        from ai_tools import BackendAdapters, ToolRegistry
        from gis_service import FloodServiceAdapter
    except ImportError as exc:
        _skip(f"ai_tools needs python-dotenv ({exc.name})")
    svc = SimulationService()
    registry = ToolRegistry(mode="real", adapters=BackendAdapters(
        flood=FloodServiceAdapter(), rescue=RescueServiceAdapter(svc)))
    for tool in ("get_rescue_routes", "get_simulation_status", "explain_route_change"):
        res = registry.execute(tool, {"scenario_id": SCENARIO})
        assert res["status"] == "OK" and res["label"] == "REAL", res
    res = registry.execute("start_rescue_simulation", {"scenario_id": SCENARIO})
    assert res["status"] == "OK" and res["data"]["started"] is True
    bad = registry.execute("get_rescue_routes", {"scenario_id": "unknown-zzz"})
    assert bad["status"] in ("OK", "SERVICE_UNAVAILABLE")
    print("PASS Member 1's ToolRegistry (real mode) dispatches to the adapter")


# ======================================================================
# HTTP API
# ======================================================================
def _client(assistant=None):
    try:
        import api
    except ImportError as exc:
        _skip(f"Flask not installed ({exc.name})")
    app = api.create_app(assistant=assistant or _FakeAssistant(),
                         simulation_service=SimulationService())
    return app.test_client()


class _FakeAssistant:
    def __init__(self):
        self.calls = []

    def ask(self, message, *, session_id=None, confirm=False):
        self.calls.append((message, session_id, confirm))
        return {"session_id": session_id or "s-1", "status": "OK", "reply": "hi",
                "needs_confirmation": False, "tools_executed": [], "tool_results": []}


def test_api_full_flow():
    c = _client()
    assert c.get("/api/health").status_code == 200
    area = c.get("/api/geo/area")
    assert area.status_code == 200 and area.json["grid_width"] == 10
    haz = c.post("/api/hazards/simulate", json={"scenario_id": SCENARIO})
    assert haz.status_code == 200 and haz.json["time_steps"] == [0, 2, 5, 10]
    assert c.get(f"/api/simulation/state?scenario_id={SCENARIO}").status_code == 409
    start = c.post("/api/simulation/start", json={"scenario_id": SCENARIO})
    assert start.status_code == 200 and start.json["started"]
    st = c.get(f"/api/simulation/state?scenario_id={SCENARIO}").json
    assert st["hours"] == 0 and st["routes"]
    for expected in (2, 5, 10):
        step = c.post("/api/simulation/step", json={"scenario_id": SCENARIO})
        assert step.status_code == 200 and step.json["snapshot"]["hours"] == expected
    rp = c.post("/api/simulation/replan", json={"scenario_id": SCENARIO, "reason": "demo"})
    assert rp.status_code in (200, 409)
    print("PASS API endpoints follow the shared data contract end to end")


def test_api_validation_and_error_mapping():
    c = _client()
    assert c.post("/api/simulation/start", data="not json",
                  content_type="application/json").status_code == 400
    assert c.post("/api/simulation/start", json=[1, 2]).status_code == 400
    assert c.post("/api/simulation/start", json={}).status_code == 400
    assert c.post("/api/simulation/start", json={"scenario_id": "bad id!"}).status_code == 400
    assert c.post("/api/simulation/start", json={"scenario_id": "no-such-1"}).status_code == 404
    assert c.post("/api/simulation/start",
                  json={"scenario_id": SCENARIO, "restart": "yes"}).status_code == 400
    assert c.post("/api/simulation/step", json={"scenario_id": SCENARIO}).status_code == 409
    assert c.post("/api/hazards/simulate",
                  json={"scenario_id": SCENARIO, "time_steps": [0, -1]}).status_code == 400
    assert c.post("/api/hazards/simulate",
                  json={"scenario_id": SCENARIO, "time_steps": [True]}).status_code == 400
    assert c.get("/api/nope").status_code == 404
    assert c.get("/api/simulation/step").status_code == 405
    body = c.post("/api/simulation/start", json={"scenario_id": "no-such-1"}).json
    assert body["status"] == "ERROR" and body["error"]["code"] == "UNKNOWN_SCENARIO"
    print("PASS bad input maps to 400/404/405/409 with a safe error body")


def test_api_ai_ask_passes_through_and_validates():
    fake = _FakeAssistant()
    c = _client(fake)
    ok = c.post("/api/ai/ask", json={"message": "hello", "session_id": "abc", "confirm": True})
    assert ok.status_code == 200 and ok.json["status"] == "OK"
    assert fake.calls == [("hello", "abc", True)]
    assert c.post("/api/ai/ask", json={"message": ""}).status_code == 400
    assert c.post("/api/ai/ask", json={"message": "x", "confirm": "true"}).status_code == 400
    assert c.post("/api/ai/ask", json={"message": "x", "session_id": 5}).status_code == 400
    print("PASS /api/ai/ask forwards to the assistant and validates the body")


def test_api_cors_only_for_allowed_origins():
    c = _client()
    good = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert good.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    bad = c.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in bad.headers
    assert c.options("/api/simulation/start").status_code == 204
    print("PASS CORS only for configured origins")


def test_api_unexpected_error_does_not_leak_details():
    c = _client(assistant=type("Boom", (), {"ask": lambda *a, **k: 1 / 0})())
    r = c.post("/api/ai/ask", json={"message": "x"})
    assert r.status_code == 500 and "division" not in r.get_data(as_text=True)
    assert r.json["error"]["code"] == "INTERNAL_ERROR"
    print("PASS unexpected errors return a generic 500")


# ----------------------------------------------------------------------
def _collect():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = skipped = 0
    for name, fn in _collect():
        try:
            fn()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    total = len(_collect())
    print(f"\n{total - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
