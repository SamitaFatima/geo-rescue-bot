# RescueBot 2.0 — Member 3 (Routing & Backend) — Handoff

Member 3 owns the rescue routing, the simulation, and the HTTP backend that
Members 1 and 4 call. **Nothing owned by Member 1 or 2 was modified.**

## Files

| File | Status | Purpose |
|---|---|---|
| `risk_astar.py` | new | Risk-aware A* (cost = 1 per move + 0/1/4 for risk 0/1/2; risk 3 impassable). Same signature and return shape as the old BFS/DFS/Bidirectional, so it plugs into `RescueAgent.plan()`, `plan_with_csp()` and `algorithm_comparison`. |
| `simulation_service.py` | new | `Simulation`, `SimulationService` (thread-safe registry) and `RescueServiceAdapter` (the exact 4-method object from `ai_tools.RescueBackendAdapter`). Runs the 0 → 2 → 5 → 10 h flood, replans, explains route changes. |
| `api.py` | new | Flask backend with every endpoint in the report's data contract, validation, error mapping, CORS for the React dev server. |
| `test_routes.py` | new | 28 tests (A*, environment hook, simulation, adapter, API, plus a 120-layout fuzz test). |
| `requirements-api.txt` | new | `flask` (the only extra dependency). |
| `environment.py` | **one additive method** | `Environment.apply_hazard_snapshot(risk_grid, protected=())` appended at the end of the class. No existing line changed. |
| `agent.py`, `csp.py`, `search_algorithms.py`, `main.py`, `algorithm_comparison.py` | unchanged | Reused as-is: `RescueCSP` picks the victim order, `plan_with_csp` drives planning, old `bfs` is kept as the comparison baseline. |

Members 1 and 2 files (`gemini_service.py`, `ai_agent.py`, `ai_tools.py`, `test_ai.py`, `geo_data.py`, `gis_service.py`, `hazard_engine.py`, `risk_analysis.py`, `test_hazards.py`, handoff docs) are byte-for-byte unchanged.

## Run it

```
pip install flask            # python-dotenv / google-genai already in requirements.txt
python api.py                # http://127.0.0.1:5000
python test_routes.py        # or: python -m pytest test_routes.py -v
```

Optional env vars: `API_HOST`, `API_PORT`, `CORS_ORIGINS`
(default allows `http://localhost:5173`, `http://localhost:3000`, `http://127.0.0.1:5173`),
plus Member 1's `AI_BACKEND_MODE` / `GEMINI_API_KEY` (keep `AI_BACKEND_MODE=mock` until the real adapters are wired and checked).

## Endpoints (shared data contract, report §4)

| Endpoint | Returns |
|---|---|
| `GET /api/geo/area` | Member 2's area info (bounds, CRS, grid, layers) |
| `POST /api/hazards/simulate` `{scenario_id, time_steps?}` | Member 2's snapshots (`risk_grid`, `blocked_cells`, `exposure`, assumptions) |
| `POST /api/simulation/start` `{scenario_id, restart?}` | `scenario_id`, `started`, `already_started`, `simulation_id`, `initial_state` |
| `GET /api/simulation/state?scenario_id=` | agents, victims, walls, hazards, routes, route_changes, events, metrics, assumptions |
| `POST /api/simulation/step` `{scenario_id}` | `snapshot` (next hour's grid) + `moves` + full `state` |
| `POST /api/simulation/replan` `{scenario_id, reason?}` | `route`, `cost`, `reason`, `changed`, `ok` |
| `POST /api/ai/ask` `{message, session_id?, confirm?}` | Member 1's assistant result, unchanged (HTTP 200 + `status`) |
| `GET /api/health` | liveness + backend mode |

Cells are always `[row, col]` (row 0 = top), the same convention as `environment.py` and Member 2's grids. Anything for the map also carries `latlon` (`{"lat","lon"}`) from `geo_data.cell_to_latlon`. Simulation and route bodies carry the "Illustrative scenario estimates, not verified forecasts." disclaimer; Member 2's bodies keep their own.

**Errors** (body `{"status":"ERROR","error":{"code","message"}}`): `400` bad JSON/field, `404` unknown scenario/endpoint, `405` wrong method, `409` simulation not started or already finished, `500` unexpected (generic text only), `503` assistant unavailable (e.g. Gemini not configured).

### State shape for Member 4 (abridged)

```json
{
  "scenario_id": "riverside-flood-01", "status": "RUNNING", "hours": 5,
  "step_index": 2, "total_steps": 4, "has_next_step": true,
  "agents": [{"agent_id": "robot-1", "position": [5,3], "latlon": {...}, "state": "SEARCHING",
              "energy": 80, "trail": [[9,9], ...], "fail_reason": null}],
  "victims": [{"position": [2,3], "latlon": {...}, "priority": "normal", "rescued": false, "risk": 1}],  // rescued victims stay listed
  "walls": [[0,5], ...],
  "hazards": {"hours": 5, "risk_grid": [[0,0,3,...]], "blocked_cells": [[0,2], ...], "exposure": {...}},
  "routes": [{"route_id": "route-1", "status": "OPEN", "target": [2,3], "path": [[5,3],[4,3],...],
              "path_latlon": [...], "cost": 7, "risk_cells": {...}, "victim_order": [[2,3]]}],
  "route_changes": [{"event_id": 12, "hours": 10, "reason": "Previous route to [2, 3] crossed flooded cell(s) ..."}],
  "metrics": {"victims_rescued": 2, "victims_total": 3, "route_changes": 1,
              "last_comparison": {"risk_astar": {...}, "bfs": {...}}}
}
```

Route `status`: `OPEN`, `BLOCKED`, `NONE`, `COMPLETE`, `FAILED`. Simulation `status`: `READY`, `RUNNING`, `DONE`, `FAILED`.

## How time works

`step` moves to the next snapshot (0 → 2 → 5 → 10 h), applies that flood to the grid, **re-plans**, then moves the robot `2 cells × hours elapsed`. Steps after hour 10 move 10 cells with the flood unchanged, until the mission ends. Calling `step` on a finished mission returns `advanced: false`.

## Demo scenario (matches report §6)

The robot starts at (9,9), exit (9,5), three victims near the river — (5,3) critical, (5,4), (2,3). Steps 1–2 (2 h, 5 h) confirm the routes; the 10 h flood blocks cells (3,3) and (4,3) on the way to the last victim, so the robot detours via column 4 and finishes. `POST /api/ai/ask` "Why did the route change?" → Member 1's assistant calls `explain_route_change`, which returns the real reason string. The layout is the `LAYOUT` constant at the top of `simulation_service.py` (walls, start, exit, victims) — edit it to change the demo.

The `metrics.last_comparison` block scores the chosen route against what the old BFS would have done on the same leg (cost on the same risk scale, medium-risk cells entered, nodes explored) — use it for the "existing algorithms vs new risk-aware A*" slide.

## Wiring into Member 1 (backend owner: this is already done in `api.py`)

```python
service = SimulationService()
adapters = BackendAdapters(flood=FloodServiceAdapter(), rescue=RescueServiceAdapter(service))
registry = create_tool_registry_from_env(adapters=adapters)
assistant = AIAssistant(gemini_service=GeminiService(), tool_registry=registry)
```

`create_app()` builds this lazily on the first `/api/ai/ask`, passing the **same** `SimulationService` to both the HTTP routes and the adapter, so the chat and the dashboard always see one simulation. If Gemini is not configured the rest of the API still works and `/api/ai/ask` returns `503 ASSISTANT_UNAVAILABLE`.

Adapter guarantees (Member 1's checklist): `start_rescue_simulation` is **idempotent** (a duplicate call returns the running simulation with `already_started: true`; only `restart: true` on the HTTP endpoint resets); read-only methods never start a simulation (they return `started: false` / `NOT_STARTED`); unknown scenario IDs raise, which `ToolRegistry` converts to `SERVICE_UNAVAILABLE`; every result is strict JSON. To switch the assistant from mock to real data set `AI_BACKEND_MODE=real` in the deployment environment (not in committed files) once the smoke tests pass.

## Limitations (say these in the pitch)

- One prepared 10×10 area; the river (column 2) is always blocked, so the mission is east of it.
- Simulations live in process memory (single worker; restart loses state).
- The existing `RescueAgent.rescue()` picks up any victim it walks over, so a normal victim on the way may be rescued before the critical one; the CSP still plans the critical victim first.
- If the flood cuts off a victim completely, the CSP proves the mission infeasible and the simulation ends `FAILED` with the reason (no silent fallback).
- All numbers are illustrative scenario estimates, not verified forecasts.
