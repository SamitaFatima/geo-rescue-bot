"""
hazard_engine.py - RescueBot 2.0 (Member 2: GIS & Hazards)

WHAT THIS FILE DOES
--------------------
Turns the static layers from geo_data.py into a TIME-STEPPED flood
simulation: "how risky is each grid cell after N hours?"

The team report asks for snapshots at 0, 2, 5 and 10 simulated hours,
with a shared risk scale:

    0 = safe
    1 = low
    2 = medium
    3 = high / blocked

The model is intentionally simple (a straight-line "rising water
level" that spreads out from the river and loses strength with
distance and elevation) - it is meant to produce a believable,
REPRODUCIBLE demo, not a real hydrology forecast. Every output is
labeled accordingly.

KEY GUARANTEE: for the same scenario_id and the same hours, you will
always get back the exact same grid. Nothing here calls random() with
a time-based seed, so results are safe to use in automated tests and
safe to show twice in a live demo without surprises.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import geo_data

# ---------------------------------------------------------------------
# The flood model's tunable numbers. Change these to make floods grow
# faster/slower or spread further - everything else adapts automatically.
# ---------------------------------------------------------------------
GROWTH_RATE_M_PER_HOUR = 0.40   # how fast the water level rises each hour
MAX_WATER_LEVEL_M = 4.0         # the water level never rises past this
DISTANCE_DECAY_M_PER_CELL = 0.15  # water "loses strength" moving away from the river

# Risk thresholds, in meters of (effective water level - ground elevation)
HIGH_RISK_MARGIN_M = 1.5
MEDIUM_RISK_MARGIN_M = 0.5
LOW_RISK_MARGIN_M = -0.5

RISK_SAFE, RISK_LOW, RISK_MEDIUM, RISK_HIGH = 0, 1, 2, 3
RISK_LABELS = {RISK_SAFE: "SAFE", RISK_LOW: "LOW", RISK_MEDIUM: "MEDIUM", RISK_HIGH: "HIGH"}

DEFAULT_TIME_STEPS: tuple[int, ...] = (0, 2, 5, 10)

DISCLAIMER = "Illustrative scenario estimate, not a verified flood forecast."


def water_level(hours: int) -> float:
    """How high the river has risen after `hours` simulated hours."""
    return min(MAX_WATER_LEVEL_M, GROWTH_RATE_M_PER_HOUR * hours)


def _risk_for_cell(elevation_m: float, distance_cells: int, hours: int) -> int:
    level = water_level(hours)
    effective_level = level - DISTANCE_DECAY_M_PER_CELL * distance_cells
    margin = effective_level - elevation_m
    if margin >= HIGH_RISK_MARGIN_M:
        return RISK_HIGH
    if margin >= MEDIUM_RISK_MARGIN_M:
        return RISK_MEDIUM
    if margin >= LOW_RISK_MARGIN_M:
        return RISK_LOW
    return RISK_SAFE


def compute_risk_grid(scenario_id: str, hours: int) -> dict[str, Any]:
    """Computes one risk grid (a 2D list of 0-3 ints) for one point in time.

    Raises geo_data.UnknownScenarioError if scenario_id is not a
    prepared scenario. Raises ValueError if hours is negative.
    """
    if hours < 0:
        raise ValueError("hours must be 0 or a positive integer.")
    scenario = geo_data.get_scenario(scenario_id)  # raises if unknown
    layers = geo_data.build_static_layers(scenario)

    risk_grid = [[RISK_SAFE] * geo_data.GRID_WIDTH for _ in range(geo_data.GRID_HEIGHT)]
    blocked_cells: list[list[int]] = []
    for r in range(geo_data.GRID_HEIGHT):
        for c in range(geo_data.GRID_WIDTH):
            if layers.river_mask[r][c]:
                risk = RISK_HIGH
            else:
                risk = _risk_for_cell(
                    layers.elevation_m[r][c],
                    layers.distance_to_river_cells[r][c],
                    hours,
                )
            risk_grid[r][c] = risk
            if risk == RISK_HIGH:
                blocked_cells.append([r, c])

    source_timestamp = _timestamp_for(scenario, hours)
    confidence = _confidence_for(hours)

    return {
        "scenario_id": scenario.scenario_id,
        "hours": hours,
        "risk_grid": risk_grid,
        "blocked_cells": blocked_cells,
        "grid_width": geo_data.GRID_WIDTH,
        "grid_height": geo_data.GRID_HEIGHT,
        "source_timestamp": source_timestamp,
        "confidence": confidence,
        "assumptions": _assumptions(),
        "disclaimer": DISCLAIMER,
    }


def get_snapshots(
    scenario_id: str, time_steps: tuple[int, ...] = DEFAULT_TIME_STEPS
) -> dict[str, Any]:
    """Multi-hour bundle matching the shared contract for
    POST /api/hazards/simulate: scenario_id, time_steps, a risk-grid
    snapshot per step, the blocked cells per step, assumptions, and a
    simple exposure (people-affected) count per step.
    """
    scenario = geo_data.get_scenario(scenario_id)  # raises if unknown
    layers = geo_data.build_static_layers(scenario)

    snapshots = []
    for hours in time_steps:
        step = compute_risk_grid(scenario_id, hours)
        exposure = _exposure_for_grid(step["risk_grid"], layers.population_density)
        snapshots.append(
            {
                "hours": hours,
                "risk_grid": step["risk_grid"],
                "blocked_cells": step["blocked_cells"],
                "source_timestamp": step["source_timestamp"],
                "confidence": step["confidence"],
                "exposure": exposure,
            }
        )

    return {
        "scenario_id": scenario.scenario_id,
        "time_steps": list(time_steps),
        "grid_width": geo_data.GRID_WIDTH,
        "grid_height": geo_data.GRID_HEIGHT,
        "snapshots": snapshots,
        "assumptions": _assumptions(),
        "disclaimer": DISCLAIMER,
    }


def _timestamp_for(scenario: geo_data.ScenarioConfig, hours: int) -> str:
    base = datetime.fromisoformat(scenario.base_timestamp.replace("Z", "+00:00"))
    stamp = base + timedelta(hours=hours)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _confidence_for(hours: int) -> float:
    """Confidence drops the further out the forecast horizon is, never
    below 0.5 and never above 0.95."""
    return round(max(0.5, min(0.95, 0.95 - 0.03 * hours)), 2)


def _exposure_for_grid(
    risk_grid: list[list[int]], population_density: list[list[int]]
) -> dict[str, int]:
    blocked = medium = low = 0
    people_at_risk = 0
    for r, row in enumerate(risk_grid):
        for c, risk in enumerate(row):
            if risk == RISK_HIGH:
                blocked += 1
            elif risk == RISK_MEDIUM:
                medium += 1
            elif risk == RISK_LOW:
                low += 1
            if risk >= RISK_MEDIUM:
                people_at_risk += population_density[r][c]
    return {
        "blocked_cells": blocked,
        "medium_risk_cells": medium,
        "low_risk_cells": low,
        "estimated_people_at_risk": people_at_risk,
    }


def _assumptions() -> list[str]:
    return [
        f"Water rises {GROWTH_RATE_M_PER_HOUR} m/hour, capped at {MAX_WATER_LEVEL_M} m.",
        f"Each grid cell is {geo_data.CELL_SIZE_METERS} m across.",
        "Elevation and population layers are synthetic, not surveyed.",
        DISCLAIMER,
    ]
