# RescueBot Member 1 AI assistant

Member 1 owns the Gemini service boundary, conversational/session handling,
tool registry interfaces, and tests. This layer does not implement flood
analysis, GIS, rescue simulation, routes, CSP, FastAPI, or a UI.

## Environment

Use Python 3.13 and the project `.venv`. Install the dependencies listed in
`requirements.txt`. Copy `.env.example` to a local `.env` and set
`GEMINI_API_KEY` only in that ignored file or in the process environment.
Never commit or print the key. `AI_BACKEND_MODE=mock` selects explicit mock
fixtures; `AI_BACKEND_MODE=real` never falls back to mock data.

The default model is `gemini-3.8-flash`. Model access, quotas, and free-tier
eligibility depend on Google AI Studio account and region. No live call is
made by importing modules or constructing a service; `ask()` performs a model
request. Verify eligibility in AI Studio before testing a live request.
The Gemini Developer API SDK client uses its standard `v1beta` endpoint.
Sanitized request failures may include an HTTP status and a structured Google
API reason code; raw provider messages and metadata are not logged or returned.

## Offline example and tests

Run the deterministic, credential-free example:

```powershell
.\.venv\Scripts\python.exe example_ai.py
```

Run Member 1 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q test_ai.py
```

## Backend integration contracts

Members 2 and 3 should expose adapters implementing the protocols in
`ai_tools.py`. Inject those adapters into `BackendAdapters` and construct
`ToolRegistry(mode="real", adapters=...)`. Expected methods and arguments:

| Owner | Method | Inputs | Return |
|---|---|---|---|
| Member 2 | `get_flood_risk(hours, scenario_id)` | integer hours, scenario identifier | mapping with backend-computed risk and provenance |
| Member 2 | `get_affected_areas(scenario_id, hours)` | scenario identifier, integer hours | mapping with backend-computed affected-area data |
| Member 3 | `start_rescue_simulation(scenario_id)` | scenario identifier | mapping with actual start/acceptance result |
| Member 3 | `get_rescue_routes(scenario_id)` | scenario identifier | mapping with actual route data and any provenance |
| Member 3 | `get_simulation_status(scenario_id)` | scenario identifier | mapping with reported status and counts |
| Member 3 | `explain_route_change(scenario_id)` | scenario identifier | mapping with route-change events/reason |

Adapters own the contract to their team's implementation. Return structured
data; do not derive missing metrics in this assistant layer. Exceptions and
missing adapters become sanitized `SERVICE_UNAVAILABLE` responses.
Each operational call needs a real `scenario_id`. The registry can receive an
explicit `default_scenario_id` from an integration that has a trusted active
scenario; otherwise the assistant asks the user for the scenario ID rather
than guessing one. The assistant also rejects a model-provided scenario ID that
does not match the user's supplied or trusted default ID.

## Future HTTP request/response shape

No endpoint is implemented here. A future `POST /api/ai/ask` handler can call
`AIAssistant.ask` with this request:

```json
{
  "session_id": "optional-session-id",
  "message": "Show flood risk after 10 hours.",
  "confirm": false
}
```

On the first call, omit `session_id`; the returned value identifies the
isolated follow-up session. A pending simulation start requires a later
explicit `"confirm": true` or a confirmation message in the same session.
The result is JSON-serializable:

```json
{
  "session_id": "generated-session-id",
  "status": "OK",
  "reply": "Assistant response",
  "needs_confirmation": false,
  "tools_executed": ["get_flood_risk"],
  "backend_mode": "mock",
  "tool_results": []
}
```

Mock results are labeled `MOCK` and explicitly state that assessment,
geographic analysis, simulation, route computation, and rescue counts are
not available in mock mode. They are not operational results.
