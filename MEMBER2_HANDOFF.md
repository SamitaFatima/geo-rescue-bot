# RescueBot 2.0 — Member 2 (GIS & Hazards) — Handoff

## What your job is, in plain words

Your friend's file (`MEMBER1_INTEGRATION_HANDOFF.md`, inside the zip she
sent you) and the team's revised report both say the same thing: **you
build the flood/hazard data, nobody else touches it.** You do NOT write
routing, the chatbot, or the dashboard — those are Members 3 and 4.

Your job has two outputs:
1. A **prepared area** (a small 10x10 map) with real-looking terrain.
2. A **flood simulation** that says, for any point in time, which
   grid cells are safe, risky, or completely blocked by water.

## The 5 files I wrote for you

| File | What it does |
|---|---|
| `geo_data.py` | Describes the one prepared area (bounds, grid size, a river, population) and the list of valid scenario IDs. |
| `hazard_engine.py` | The actual flood model — turns "how many hours have passed" into a grid of risk values (0 = safe, 1 = low, 2 = medium, 3 = high/blocked). |
| `risk_analysis.py` | Turns that raw grid into the two things your teammates actually need: one overall risk level, and a short list of 4 named flooded zones. |
| `gis_service.py` | **The file Member 1 and whoever builds the backend will import.** It wraps everything above into the exact adapter class your friend's `ai_tools.py` already expects, called `FloodServiceAdapter`. |
| `test_hazards.py` | 17 automated checks, including a real test against a copy of your friend's `ai_tools.py` — all 17 currently pass. |

I ran all of this myself before sending it to you — it works out of the box.

## Where to put these files

Drop all 5 files **into the same project folder** as the files your
friend already sent you (`environment.py`, `agent.py`, `ai_tools.py`,
`ai_agent.py`, `gemini_service.py`, etc). Nothing needs renaming, and
you don't need to edit any of her files.

## How your part connects to hers (already working, no setup needed)

Her `ai_tools.py` says Member 2 must build an object with exactly these
two methods:

```python
def get_flood_risk(self, hours: int, scenario_id: str) -> Mapping[str, Any]: ...
def get_affected_areas(self, scenario_id: str, hours: int) -> Mapping[str, Any]: ...
```

`gis_service.py` has exactly that, in a class called `FloodServiceAdapter`.
Whoever wires the whole backend together (per the report, that's usually
Member 3) will do this — you don't have to write this part, just know
it exists:

```python
from ai_tools import BackendAdapters, create_tool_registry_from_env
from gis_service import FloodServiceAdapter

adapters = BackendAdapters(
    flood=FloodServiceAdapter(),      # <- your part, already done
    rescue=RescueServiceAdapter(...), # <- Member 3's part
)
registry = create_tool_registry_from_env(adapters=adapters)
```

## How to run the tests yourself

1. Put all 5 files in the project folder (next to `ai_tools.py`, `requirements.txt`, etc).
2. Open a terminal in that folder.
3. Install the one thing you need (if you don't already have it):
   ```
   pip install pytest
   ```
4. Run:
   ```
   python -m pytest test_hazards.py -v
   ```
   You should see **17 passed**.

## Try it yourself in 30 seconds (optional, but satisfying)

In the same folder, run `python`, then:

```python
import hazard_engine
bundle = hazard_engine.get_snapshots("riverside-flood-01")
for s in bundle["snapshots"]:
    print(s["hours"], "hours ->", s["exposure"])
```

You'll see the flood getting worse hour by hour — that's your demo moment
("Show flood risk after 10 hours").

## The prepared scenario

- `scenario_id = "riverside-flood-01"` — the main one, use this everywhere.
- `scenario_id = "scenario-17"` — a second name for the exact same
  scenario, kept only because your friend's handoff doc used that
  name in its examples. Either works.
- Area: a made-up 10x10 grid called "Riverside District," with a river
  running down one side. This matches the report's instruction: "one
  prepared geographical area... not a nationwide prediction system."
- Every output is labeled with: *"Illustrative scenario estimate, not a
  verified flood forecast."* — the report specifically asked for this
  exact kind of disclaimer.

## What to tell Member 3 and Member 4 (your handoff to them)

- **Member 3** (routing): call `gis_service.GISService().get_hazard_snapshot("riverside-flood-01")`
  to get the 0/2/5/10-hour risk grids, plus a `blocked_cells` list per
  time step, in the same (row, col) coordinates as the existing
  `environment.py` grid — no translation needed.
- **Member 4** (dashboard): call `gis_service.GISService().get_area_info()`
  for the map bounds, and use `geo_data.cell_to_latlon(row, col)` to
  place any grid cell on an actual map.

## If you want to change anything

- **Make the flood spread faster/slower, or reach further**: edit the
  three numbers at the top of `hazard_engine.py` (`GROWTH_RATE_M_PER_HOUR`,
  `MAX_WATER_LEVEL_M`, `DISTANCE_DECAY_M_PER_CELL`). Bigger numbers =
  faster/wider flooding.
- **Add a second scenario**: add one entry to the `SCENARIOS` dictionary
  in `geo_data.py` — copy the existing one and change the `scenario_id`
  and `terrain_seed`.
- **Rename the area or the 4 zones**: edit `AREA_INFO` in `geo_data.py`
  and `_zones_for_grid()` in `risk_analysis.py`.

No new libraries are needed — everything is written in plain Python, so
it won't conflict with anything your teammates installed.
