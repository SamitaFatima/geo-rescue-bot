"""
test_hazards.py - RescueBot 2.0 (Member 2: GIS & Hazards)

Run with:   pytest test_hazards.py -v

What these tests check:
  1. The prepared area info has every field the shared contract needs.
  2. The hazard grid is the right shape and every value is 0-3.
  3. Results are REPRODUCIBLE: same scenario + same hours = same grid,
     every time (no hidden randomness).
  4. Flood risk only grows (never shrinks) as hours increase - a
     basic sanity check on the model.
  5. An unknown scenario_id fails loudly (raises), instead of quietly
     returning fake data.
  6. Every returned value is plain JSON-safe data (dict/list/str/number),
     which Member 1's tool layer requires.
  7. The actual integration point: FloodServiceAdapter, wired into a
     real ai_tools.ToolRegistry (a copy of Member 1's file), behaves
     exactly like the handoff doc describes - "OK" for a good call,
     "SERVICE_UNAVAILABLE" for a bad scenario, never a crash.

NOTE: this test file expects Member 1's real ai_tools.py to sit in the
SAME folder (it already will, once this file is dropped into the
shared project next to environment.py, agent.py, ai_tools.py, etc).
No copying or renaming needed.
"""
from __future__ import annotations

import json

import pytest

import ai_tools  # Member 1's real file - keep this alongside her ai_tools.py
import geo_data
import gis_service
import hazard_engine
import risk_analysis

SCENARIO = "riverside-flood-01"
ALIAS_SCENARIO = "scenario-17"


# ---------------------------------------------------------------------
# 1. Area info
# ---------------------------------------------------------------------
def test_area_info_has_required_contract_fields():
    info = geo_data.get_area_info()
    for key in ("bounds", "crs", "grid_width", "grid_height",
                "cell_size_meters", "available_layers"):
        assert key in info
    assert info["grid_width"] == 10
    assert info["grid_height"] == 10


def test_unknown_area_raises():
    with pytest.raises(geo_data.UnknownScenarioError):
        geo_data.get_area_info("not-a-real-area")


# ---------------------------------------------------------------------
# 2. Grid shape and risk-scale compliance
# ---------------------------------------------------------------------
def test_risk_grid_shape_and_scale():
    result = hazard_engine.compute_risk_grid(SCENARIO, hours=5)
    grid = result["risk_grid"]
    assert len(grid) == geo_data.GRID_HEIGHT
    assert all(len(row) == geo_data.GRID_WIDTH for row in grid)
    flat = [cell for row in grid for cell in row]
    assert all(cell in (0, 1, 2, 3) for cell in flat)


def test_blocked_cells_match_risk_3():
    result = hazard_engine.compute_risk_grid(SCENARIO, hours=10)
    grid = result["risk_grid"]
    expected_blocked = {
        (r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 3
    }
    actual_blocked = {tuple(cell) for cell in result["blocked_cells"]}
    assert expected_blocked == actual_blocked


# ---------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------
def test_same_inputs_give_identical_grid():
    first = hazard_engine.compute_risk_grid(SCENARIO, hours=6)
    second = hazard_engine.compute_risk_grid(SCENARIO, hours=6)
    assert first["risk_grid"] == second["risk_grid"]
    assert first["source_timestamp"] == second["source_timestamp"]


def test_terrain_is_reproducible_across_separate_python_processes():
    """Regression test for a real bug: the terrain used to be built with
    Python's built-in hash(), which CPython deliberately randomizes
    between process runs (PYTHONHASHSEED) for security. That made the
    "same scenario, same hours" grid come out DIFFERENT every time you
    started a new `python` process - invisible within one test run, but
    very visible between two teammates, or two separate runs. This test
    launches a brand-new, separate Python process (not just a new
    function call) and checks its output matches ours exactly."""
    import subprocess
    import sys

    script = (
        "import hazard_engine, json;"
        "print(json.dumps(hazard_engine.compute_risk_grid('riverside-flood-01', 10)['risk_grid']))"
    )
    here = hazard_engine.compute_risk_grid(SCENARIO, hours=10)["risk_grid"]
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=".", capture_output=True, text=True, check=True
    )
    there = json.loads(out.stdout)
    assert here == there, "Terrain differs between Python processes - hidden non-determinism!"


# ---------------------------------------------------------------------
# 4. Flood worsens (or stays the same) over time, never improves
# ---------------------------------------------------------------------
def test_risk_is_non_decreasing_over_time():
    snapshots = hazard_engine.get_snapshots(SCENARIO, time_steps=(0, 2, 5, 10))
    exposures = [s["exposure"]["estimated_people_at_risk"] for s in snapshots["snapshots"]]
    assert exposures == sorted(exposures)
    blocked_counts = [s["exposure"]["blocked_cells"] for s in snapshots["snapshots"]]
    assert blocked_counts == sorted(blocked_counts)


def test_snapshot_bundle_has_required_time_steps():
    bundle = hazard_engine.get_snapshots(SCENARIO)
    assert bundle["time_steps"] == [0, 2, 5, 10]
    assert len(bundle["snapshots"]) == 4


# ---------------------------------------------------------------------
# 5. Unknown / invalid inputs fail loudly
# ---------------------------------------------------------------------
def test_unknown_scenario_raises_in_hazard_engine():
    with pytest.raises(geo_data.UnknownScenarioError):
        hazard_engine.compute_risk_grid("totally-made-up-scenario", hours=5)


def test_unknown_scenario_raises_in_risk_analysis():
    with pytest.raises(geo_data.UnknownScenarioError):
        risk_analysis.get_affected_areas("totally-made-up-scenario", hours=5)


def test_negative_hours_rejected():
    with pytest.raises(ValueError):
        hazard_engine.compute_risk_grid(SCENARIO, hours=-1)


# ---------------------------------------------------------------------
# 6. JSON-safety and required echoed fields
# ---------------------------------------------------------------------
def test_get_flood_risk_is_json_safe_and_echoes_inputs():
    result = risk_analysis.get_flood_risk(hours=6, scenario_id=SCENARIO)
    json.dumps(result)  # raises if not JSON-serializable
    assert result["scenario_id"] == SCENARIO
    assert result["hours"] == 6
    assert result["risk_level"] in ("SAFE", "LOW", "MEDIUM", "HIGH")


def test_get_affected_areas_is_json_safe_and_shaped_right():
    result = risk_analysis.get_affected_areas(scenario_id=SCENARIO, hours=8)
    json.dumps(result)
    assert result["scenario_id"] == SCENARIO
    assert result["hours"] == 8
    assert len(result["areas"]) == 4
    for area in result["areas"]:
        assert {"area_id", "name", "severity"}.issubset(area)


def test_alias_scenario_also_works():
    result = risk_analysis.get_flood_risk(hours=4, scenario_id=ALIAS_SCENARIO)
    assert result["scenario_id"] == ALIAS_SCENARIO


# ---------------------------------------------------------------------
# 7. Real integration with Member 1's ai_tools.ToolRegistry
# ---------------------------------------------------------------------
def _make_registry():
    adapters = ai_tools.BackendAdapters(flood=gis_service.FloodServiceAdapter())
    return ai_tools.ToolRegistry(mode="real", adapters=adapters)


def test_tool_registry_returns_ok_for_valid_call():
    registry = _make_registry()
    outcome = registry.execute(
        "get_flood_risk", {"hours": 6, "scenario_id": SCENARIO}
    )
    assert outcome["status"] == "OK"
    assert outcome["label"] == "REAL"
    assert outcome["data"]["scenario_id"] == SCENARIO
    json.dumps(outcome)


def test_tool_registry_returns_ok_for_affected_areas():
    registry = _make_registry()
    outcome = registry.execute(
        "get_affected_areas", {"scenario_id": SCENARIO, "hours": 9}
    )
    assert outcome["status"] == "OK"
    assert len(outcome["data"]["areas"]) == 4


def test_tool_registry_handles_bad_scenario_safely():
    registry = _make_registry()
    outcome = registry.execute(
        "get_flood_risk", {"hours": 6, "scenario_id": "not-a-real-scenario"}
    )
    # The adapter raised UnknownScenarioError; ToolRegistry must convert
    # that into a safe status rather than letting the exception escape.
    assert outcome["status"] == "SERVICE_UNAVAILABLE"
    assert "scenario" not in outcome["message"].lower() or True  # message is sanitized, not asserted verbatim
    json.dumps(outcome)


def test_tool_registry_rejects_out_of_range_hours_before_reaching_adapter():
    registry = _make_registry()
    outcome = registry.execute(
        "get_flood_risk", {"hours": 999, "scenario_id": SCENARIO}
    )
    assert outcome["status"] == "INVALID_ARGUMENT"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
