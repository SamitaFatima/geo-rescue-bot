"""
geo_data.py - RescueBot 2.0 (Member 2: GIS & Hazards)

WHAT THIS FILE DOES
--------------------
Defines the ONE prepared geographical area used for the hackathon demo
(the team report says: "one prepared geographical area and a small
aligned grid, not a nationwide prediction system"), and builds the
static map layers that do not change over time: elevation, how close
each cell is to the river, and population density.

Everything here is synthetic (made up, but reproducible) because no
real GIS data source was verified for this hackathon. That is fine -
the team report explicitly asks for "illustrative scenario estimates,
not verified forecasts."

HOW IT LINES UP WITH THE REST OF THE TEAM
------------------------------------------
- The grid uses the SAME (row, col) convention as the existing
  environment.py (row 0 = top, col 0 = left), so Member 3's routing
  code can line a hazard cell up with an agent cell directly.
- GRID_WIDTH / GRID_HEIGHT default to 10 x 10, matching the existing
  environment.py grid size, so Member 3 does not have to rescale
  anything.
- get_area_info() returns exactly the fields the shared data contract
  asks for from GET /api/geo/area: bounds, a CRS/reference note, grid
  dimensions, cell size, and the available layers.

WHERE TO EDIT IF YOU WANT TO CHANGE THE SCENARIO
--------------------------------------------------
- To rename or re-describe the area: edit AREA_INFO below.
- To add a new scenario: add one entry to SCENARIOS. Keep the
  scenario_id to letters, digits, "_" and "-" only (this matches the
  validation rule Member 1's ai_tools.py already enforces).
- To change how "risky" an area looks (more or less river, more or
  less population) edit the numbers inside build_static_layers().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------
# The prepared area
# ---------------------------------------------------------------------
AREA_ID = "riverside-district-01"

# Grid size matches the existing environment.py (10x10) so Member 3's
# routing / agent code can reuse row,col coordinates directly.
GRID_WIDTH = 10
GRID_HEIGHT = 10
CELL_SIZE_METERS = 50  # each grid cell represents a 50m x 50m patch of land

# A made-up bounding box (not a real surveyed location). CRS = the
# coordinate system used for the lat/lon numbers below.
AREA_BOUNDS = {
    "min_lat": 24.8600,
    "max_lat": 24.8690,
    "min_lon": 67.0100,
    "max_lon": 67.0190,
    "crs": "EPSG:4326 (illustrative placement only, not a surveyed boundary)",
}

AVAILABLE_LAYERS = [
    "elevation_m",          # synthetic ground height above the river, in meters
    "distance_to_river_cells",  # how many grid cells away from the river
    "population_density",   # synthetic people-per-cell, for exposure estimates
    "river_mask",            # 1 where the river itself is, else 0
]

# River runs down column 2 of the grid (a simple, readable layout for a demo).
RIVER_COLUMNS = (2,)

AREA_INFO: dict[str, Any] = {
    "area_id": AREA_ID,
    "name": "Riverside District (illustrative demo area)",
    "bounds": AREA_BOUNDS,
    "crs": AREA_BOUNDS["crs"],
    "grid_width": GRID_WIDTH,
    "grid_height": GRID_HEIGHT,
    "cell_size_meters": CELL_SIZE_METERS,
    "available_layers": AVAILABLE_LAYERS,
    "disclaimer": "Illustrative scenario area, not a verified GIS survey.",
}


# ---------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    area_id: str
    description: str
    # ISO timestamp used as "hour 0" for this scenario, so that every
    # reported source_timestamp is reproducible instead of using the
    # real wall-clock (which would make results different every run).
    base_timestamp: str = "2026-10-23T00:00:00Z"
    # A small integer that seeds the synthetic terrain so the same
    # scenario always generates the same elevation/population layers.
    terrain_seed: int = 1


SCENARIOS: dict[str, ScenarioConfig] = {
    "riverside-flood-01": ScenarioConfig(
        scenario_id="riverside-flood-01",
        area_id=AREA_ID,
        description="Primary demo scenario: river overflow across the prepared area.",
        terrain_seed=1,
    ),
    # Registered as an alias so the exact example scenario_id used in
    # Member 1's integration handoff ("scenario-17") also works out of
    # the box during the team demo.
    "scenario-17": ScenarioConfig(
        scenario_id="scenario-17",
        area_id=AREA_ID,
        description="Alias of riverside-flood-01 (kept for handoff-doc compatibility).",
        terrain_seed=1,
    ),
}


class UnknownScenarioError(KeyError):
    """Raised when a scenario_id is not one of the prepared, trusted scenarios."""


def get_scenario(scenario_id: str) -> ScenarioConfig:
    """Look up a prepared scenario. Raises UnknownScenarioError if not found.

    This is deliberately a hard failure (not a silent fallback): the
    team report requires that an invalid scenario never silently
    returns made-up data.
    """
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise UnknownScenarioError(
            f"'{scenario_id}' is not a prepared/trusted scenario."
        ) from exc


def get_area_info(area_id: str = AREA_ID) -> dict[str, Any]:
    """Matches GET /api/geo/area in the shared data contract."""
    if area_id != AREA_ID:
        raise UnknownScenarioError(f"'{area_id}' is not a prepared area.")
    return dict(AREA_INFO)


# ---------------------------------------------------------------------
# Coordinate helpers (grid <-> lat/lon), used by Member 4's map
# ---------------------------------------------------------------------
def cell_to_latlon(row: int, col: int) -> dict[str, float]:
    """Approximate lat/lon for the CENTER of a grid cell, for map alignment.

    This is a simple straight-line (linear) mapping across the bounding
    box - good enough for a small illustrative area, not a real
    projection.
    """
    if not (0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH):
        raise ValueError(f"({row}, {col}) is outside the {GRID_HEIGHT}x{GRID_WIDTH} grid.")
    lat_span = AREA_BOUNDS["max_lat"] - AREA_BOUNDS["min_lat"]
    lon_span = AREA_BOUNDS["max_lon"] - AREA_BOUNDS["min_lon"]
    # row 0 is the top of the grid -> the northern (higher-latitude) edge
    lat = AREA_BOUNDS["max_lat"] - (row + 0.5) / GRID_HEIGHT * lat_span
    lon = AREA_BOUNDS["min_lon"] + (col + 0.5) / GRID_WIDTH * lon_span
    return {"lat": round(lat, 6), "lon": round(lon, 6)}


# ---------------------------------------------------------------------
# Static layers (do not change over time)
# ---------------------------------------------------------------------
@dataclass
class StaticLayers:
    elevation_m: list[list[float]]
    distance_to_river_cells: list[list[int]]
    population_density: list[list[int]]
    river_mask: list[list[int]]


def _deterministic_unit(a: int, b: int, seed: int) -> float:
    """A reproducible pseudo-random number in [0, 1) for cell (a, b).

    IMPORTANT: this must NOT use Python's built-in hash() on strings or
    tuples containing strings. CPython randomizes that hash differently
    every time you start python (PYTHONHASHSEED), as a security feature -
    so hash() gives a different "random" number on every run, on every
    machine. That silently breaks reproducibility between teammates and
    between test runs. hashlib.sha256 has no such randomization: the same
    input bytes always produce the same digest, on any machine, forever.
    """
    import hashlib

    key = f"rescuebot-geo:{seed}:{a}:{b}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:4], "big")
    return (value % 10_000) / 10_000.0


def build_static_layers(scenario: ScenarioConfig) -> StaticLayers:
    """Builds the elevation, river, distance-to-river, and population
    layers for one scenario's terrain seed. Deterministic: calling this
    twice with the same scenario always returns identical layers.
    """
    seed = scenario.terrain_seed
    river_mask = [
        [1 if c in RIVER_COLUMNS else 0 for c in range(GRID_WIDTH)]
        for r in range(GRID_HEIGHT)
    ]

    # Elevation rises smoothly as you move away from the river columns,
    # plus a small deterministic "bump" so the terrain is not perfectly
    # flat/uniform.
    elevation_m = [[0.0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            dist_to_river_col = min(abs(c - rc) for rc in RIVER_COLUMNS)
            base = dist_to_river_col * 1.2
            bump = _deterministic_unit(r, c, seed) * 1.5
            elevation_m[r][c] = 0.0 if river_mask[r][c] else round(base + bump, 2)

    distance_to_river_cells = _bfs_distance_to_mask(river_mask)

    # Population density: a bit higher near the center of the grid
    # (a stand-in "town center"), lower near the river itself.
    center_r, center_c = GRID_HEIGHT / 2, GRID_WIDTH / 2
    population_density = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if river_mask[r][c]:
                population_density[r][c] = 0
                continue
            dist_from_center = math.hypot(r - center_r, c - center_c)
            density = max(5, round(40 - dist_from_center * 4
                                    + _deterministic_unit(r, c, seed + 7) * 10))
            population_density[r][c] = int(density)

    return StaticLayers(
        elevation_m=elevation_m,
        distance_to_river_cells=distance_to_river_cells,
        population_density=population_density,
        river_mask=river_mask,
    )


def _bfs_distance_to_mask(mask: list[list[int]]) -> list[list[int]]:
    """Breadth-first distance (in grid steps) from the nearest mask==1 cell."""
    from collections import deque

    height, width = len(mask), len(mask[0])
    dist = [[-1] * width for _ in range(height)]
    q: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in range(width):
            if mask[r][c]:
                dist[r][c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist
