"""
risk_analysis.py - RescueBot 2.0 (Member 2: GIS & Hazards)

WHAT THIS FILE DOES
--------------------
Takes the raw risk grid from hazard_engine.py (a 2D list of 0-3 numbers)
and turns it into the two things the rest of the team actually wants to
read:

1. get_flood_risk(hours, scenario_id)
   -> ONE overall risk summary for the whole area (used by the "Show
      flood risk after 10 hours" voice/chat question in the demo).

2. get_affected_areas(scenario_id, hours)
   -> a short list of named zones ("North District", etc.) with a
      severity label and an estimated number of people affected, so
      the dashboard can show something more readable than a raw grid.

These two function signatures match FloodBackendAdapter in Member 1's
ai_tools.py EXACTLY (same parameter names, same order), so they can be
wired in directly - see gis_service.py, which does exactly that.
"""
from __future__ import annotations

from typing import Any

import geo_data
import hazard_engine

RISK_LABELS = hazard_engine.RISK_LABELS


# ---------------------------------------------------------------------
# Zones: the 10x10 grid is split into four named quadrants. This keeps
# the "affected areas" list short and readable for a demo, instead of
# showing all 100 individual cells.
# ---------------------------------------------------------------------
def _zones_for_grid(height: int, width: int) -> list[dict[str, Any]]:
    mid_r, mid_c = height // 2, width // 2
    return [
        {
            "area_id": "zone-nw",
            "name": "Northwest District",
            "rows": range(0, mid_r),
            "cols": range(0, mid_c),
        },
        {
            "area_id": "zone-ne",
            "name": "Northeast District",
            "rows": range(0, mid_r),
            "cols": range(mid_c, width),
        },
        {
            "area_id": "zone-sw",
            "name": "Southwest District",
            "rows": range(mid_r, height),
            "cols": range(0, mid_c),
        },
        {
            "area_id": "zone-se",
            "name": "Southeast District",
            "rows": range(mid_r, height),
            "cols": range(mid_c, width),
        },
    ]


def _summarize_zone(
    zone: dict[str, Any],
    risk_grid: list[list[int]],
    population_density: list[list[int]],
) -> dict[str, Any]:
    max_risk = hazard_engine.RISK_SAFE
    blocked_cells = 0
    people_affected = 0
    for r in zone["rows"]:
        for c in zone["cols"]:
            risk = risk_grid[r][c]
            max_risk = max(max_risk, risk)
            if risk == hazard_engine.RISK_HIGH:
                blocked_cells += 1
            if risk >= hazard_engine.RISK_MEDIUM:
                people_affected += population_density[r][c]
    return {
        "area_id": zone["area_id"],
        "name": zone["name"],
        "severity": RISK_LABELS[max_risk],
        "blocked_cells": blocked_cells,
        "estimated_people_affected": people_affected,
    }


def get_affected_areas(scenario_id: str, hours: int) -> dict[str, Any]:
    """Matches FloodBackendAdapter.get_affected_areas(scenario_id, hours)
    in Member 1's ai_tools.py exactly (same parameter names/order).

    Raises geo_data.UnknownScenarioError for an unrecognized scenario,
    and ValueError for a bad `hours` value - both are plain exceptions,
    which is what ai_tools.ToolRegistry expects so it can turn a failure
    into a safe SERVICE_UNAVAILABLE result instead of crashing.
    """
    scenario = geo_data.get_scenario(scenario_id)  # raises if unknown
    step = hazard_engine.compute_risk_grid(scenario_id, hours)
    layers = geo_data.build_static_layers(scenario)

    zones = _zones_for_grid(geo_data.GRID_HEIGHT, geo_data.GRID_WIDTH)
    areas = [
        _summarize_zone(zone, step["risk_grid"], layers.population_density)
        for zone in zones
    ]

    return {
        "scenario_id": scenario.scenario_id,
        "hours": hours,
        "areas": areas,
        "source_timestamp": step["source_timestamp"],
        "assumptions": step["assumptions"],
        "disclaimer": hazard_engine.DISCLAIMER,
    }


def get_flood_risk(hours: int, scenario_id: str) -> dict[str, Any]:
    """Matches FloodBackendAdapter.get_flood_risk(hours, scenario_id)
    in Member 1's ai_tools.py exactly (same parameter names/order).

    Returns one overall risk_level for the whole prepared area, plus
    enough extra detail (blocked cell count, confidence) for the
    dashboard and the AI assistant's explanation to use.
    """
    step = hazard_engine.compute_risk_grid(scenario_id, hours)
    overall_risk = max((max(row) for row in step["risk_grid"]), default=0)

    return {
        "scenario_id": step["scenario_id"],
        "hours": hours,
        "risk_level": RISK_LABELS[overall_risk],
        "confidence": step["confidence"],
        "blocked_cells": len(step["blocked_cells"]),
        "grid_width": step["grid_width"],
        "grid_height": step["grid_height"],
        "source_timestamp": step["source_timestamp"],
        "assumptions": step["assumptions"],
        "disclaimer": hazard_engine.DISCLAIMER,
    }
