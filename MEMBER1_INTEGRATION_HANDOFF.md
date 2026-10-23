# RescueBot 2.0 — Member 1 Integration Handoff

## Scope and current state

Member 1 provides the Gemini boundary, conversational session manager, and the validated tool-dispatch boundary between Gemini and the hazard/rescue services.

- Gemini model: `gemini-3.8-flash`
- Safe default backend mode: `mock`
- Six allowlisted tools; arbitrary model-requested functions are rejected
- Explicit confirmation is required before `start_rescue_simulation`
- Scenario IDs and forecast horizons are validated before an adapter is called
- Gemini/provider error messages are sanitized before they reach logs or clients
- Mock responses are labeled `MOCK` and do not claim real forecasts, routes, or rescue counts

Member 1 does **not** implement HTTP routing, flood/GIS analysis, rescue simulation, persistence, or the frontend.

## Files and ownership

| File | Purpose |
|---|---|
| `gemini_service.py` | Loads Gemini configuration, constructs SDK requests, applies bounded retries, and translates provider/network errors safely. |
| `ai_agent.py` | Manages conversations, Gemini tool-call rounds, scenario context, confirmation, and the JSON-serializable chat result. |
| `ai_tools.py` | Defines tool schemas, validates arguments, exposes Member 2/3 adapter protocols, and dispatches mock or real calls. |
| `test_ai.py` | Member 1 offline contract, safety, tool-dispatch, session, confirmation, and error-handling tests. |
| `.env.example` | Non-secret configuration template. |

Keep the real `.env` local and never commit or share it.

## Member 2: hazard/flood service integration

Member 2 should implement an object matching `FloodBackendAdapter` in `ai_tools.py`. The method signatures are exact:

```python
from collections.abc import Mapping
from typing import Any

class FloodServiceAdapter:
    def get_flood_risk(
        self, hours: int, scenario_id: str
    ) -> Mapping[str, Any]:
        ...

    def get_affected_areas(
        self, scenario_id: str, hours: int
    ) -> Mapping[str, Any]:
        ...
```

Input contract:

- `scenario_id`: 1–64 characters; letters, digits, `_`, and `-` only. It must identify an existing trusted scenario.
- `hours`: an integer from 1 through 168. Booleans and numeric strings are rejected.
- Do not change parameter names or argument order in the adapter methods.

Return contract:

- Return a `Mapping[str, Any]`, normally a plain dictionary.
- Every nested value must be strict JSON data: objects, arrays, strings, finite numbers, booleans, or `null`.
- Include `scenario_id` and `hours` in results so the source context remains inspectable.
- Use domain-owned fields for real results, for example:

```json
{
  "scenario_id": "scenario-17",
  "hours": 6,
  "risk_level": "HIGH",
  "confidence": 0.84,
  "source_timestamp": "2026-10-22T10:30:00Z"
}
```

```json
{
  "scenario_id": "scenario-17",
  "hours": 6,
  "areas": [
    {"area_id": "zone-a", "name": "North Zone", "severity": "HIGH"}
  ],
  "source_timestamp": "2026-10-22T10:30:00Z"
}
```

The adapter may call Member 2 code directly or use an internal HTTP client. It should enforce its own authentication, timeout, and data-freshness policy. Raise an exception when the downstream service cannot complete the call; `ToolRegistry` converts it to a sanitized `SERVICE_UNAVAILABLE` result without logging private exception text.

## Member 3: rescue simulation integration

Member 3 should implement an object matching `RescueBackendAdapter` in `ai_tools.py`:

```python
from collections.abc import Mapping
from typing import Any

class RescueServiceAdapter:
    def start_rescue_simulation(
        self, scenario_id: str
    ) -> Mapping[str, Any]:
        ...

    def get_rescue_routes(
        self, scenario_id: str
    ) -> Mapping[str, Any]:
        ...

    def get_simulation_status(
        self, scenario_id: str
    ) -> Mapping[str, Any]:
        ...

    def explain_route_change(
        self, scenario_id: str
    ) -> Mapping[str, Any]:
        ...
```

All results must be strict JSON mappings. Suggested shapes are:

```json
{"scenario_id":"scenario-17","started":true,"simulation_id":"sim-42","state":"RUNNING"}
```

```json
{"scenario_id":"scenario-17","routes":[{"route_id":"route-1","status":"OPEN"}]}
```

```json
{"scenario_id":"scenario-17","status":"RUNNING","victims_rescued":4}
```

```json
{"scenario_id":"scenario-17","route_id":"route-1","reason":"Road segment closed by backend event data."}
```

Important behavior:

- `start_rescue_simulation` is not dispatched until the user explicitly confirms it.
- Confirmation is enforced by `AIAssistant`, but Member 3 should also make the start operation idempotent to protect against duplicate delivery or HTTP retries.
- Never start a simulation from any read-only method.
- An absent adapter in real mode returns `SERVICE_UNAVAILABLE`; it never falls back silently to mock data.

## Wiring Member 2 and Member 3 into Member 1

The team backend composition root should create the adapters once, then inject them into `ToolRegistry` and `AIAssistant`:

```python
from ai_agent import AIAssistant
from ai_tools import BackendAdapters, create_tool_registry_from_env
from gemini_service import GeminiService

adapters = BackendAdapters(
    flood=FloodServiceAdapter(...),   # Member 2
    rescue=RescueServiceAdapter(...), # Member 3
)

registry = create_tool_registry_from_env(adapters=adapters)
assistant = AIAssistant(
    gemini_service=GeminiService(),
    tool_registry=registry,
)
```

Do this in the backend startup/composition layer, not inside `ai_tools.py`. This preserves Member 1's tested boundary and keeps teammate services replaceable.

Before live mode, add integration tests that inject fake Member 2/3 adapters and assert all six methods dispatch with the expected validated arguments and JSON results.

## Member 4: frontend chat integration

Member 4 should not import `ai_agent.py`, call Gemini directly, or receive the Gemini key. The team's backend should expose an authenticated endpoint around the long-lived `assistant` instance.

Proposed endpoint:

```text
POST /api/ai/ask
Content-Type: application/json
```

Request body:

```json
{
  "message": "Check flood risk after 6 hours for scenario_id: scenario-17.",
  "session_id": "optional-session-id-from-a-previous-response",
  "confirm": false
}
```

Backend call:

```python
result = assistant.ask(
    body["message"],
    session_id=body.get("session_id"),
    confirm=body.get("confirm", False),
)
```

Typical success response:

```json
{
  "session_id": "server-generated-id",
  "status": "OK",
  "reply": "REAL result for get_flood_risk: ...",
  "needs_confirmation": false,
  "tools_executed": ["get_flood_risk"],
  "tool_results": [
    {
      "status": "OK",
      "mode": "real",
      "label": "REAL",
      "tool": "get_flood_risk",
      "data": {
        "scenario_id": "scenario-17",
        "hours": 6,
        "risk_level": "HIGH"
      },
      "message": "Result returned by the real backend."
    }
  ],
  "backend_mode": "real"
}
```

Confirmation response and follow-up:

```json
{
  "session_id": "server-generated-id",
  "status": "CONFIRMATION_REQUIRED",
  "reply": "I can request the rescue simulation for this scenario. Please confirm or cancel.",
  "needs_confirmation": true,
  "tools_executed": [],
  "tool_results": [],
  "backend_mode": "mock"
}
```

The frontend must retain the returned `session_id`. To confirm, send a new request with the same session and a boolean—not a string:

```json
{
  "message": "Proceed.",
  "session_id": "server-generated-id",
  "confirm": true
}
```

To cancel, send `"message": "cancel"`, the same `session_id`, and `"confirm": false`.

Frontend requirements:

- Treat `reply` as text; do not render it as HTML.
- Show an explicit confirmation control when `needs_confirmation` is `true`.
- Display the `MOCK`/`REAL` label clearly for operational results.
- Do not invent or replace `session_id`; discard it only when the user starts a new chat.
- Limit `message` to 4,000 characters; empty messages are invalid.
- Disable duplicate submits while a request is pending.
- Never expose provider diagnostics or secrets in the UI. A generic retry message is sufficient for Gemini availability failures.

The backend should map malformed JSON to HTTP 400, unauthenticated access to 401/403, and unexpected server failures to 500. The `AIAssistant` result itself is a domain response and may be returned as JSON with HTTP 200 so the frontend can branch on `status` consistently. The team may choose a different HTTP mapping, but it must document that mapping for Member 4.

Current sessions are held in the `AIAssistant` process memory. The same assistant instance must be reused across requests. Multiple backend workers require sticky routing or a shared conversation store before horizontal scaling; otherwise a valid `session_id` can produce `SESSION_NOT_FOUND` on another worker.

## Response statuses and error handling

Common assistant statuses:

| Status | Meaning / client action |
|---|---|
| `OK` | Render `reply`; inspect `tool_results` for operational data. |
| `CONFIRMATION_REQUIRED` | Show confirm/cancel controls and preserve `session_id`. |
| `SCENARIO_REQUIRED` | Ask for a valid scenario ID. |
| `INVALID_REQUEST` | Correct the request payload. |
| `SESSION_NOT_FOUND` | Start a new conversation session. |
| `NO_BACKEND_RESULT` | Do not present model text as verified operational data. |
| `TOOL_LIMIT_REACHED` | Ask a narrower question. |
| `INVALID_MODEL_RESPONSE` | Show a safe retry message. |
| `CONFIGURATION_ERROR` / `AUTHENTICATION_FAILED` | Backend operator must correct deployment configuration. |
| `QUOTA_EXHAUSTED` | Retry later according to team policy; do not loop in the frontend. |
| `GEMINI_TIMEOUT` / `GEMINI_UNAVAILABLE` | Temporary failure; show retry-later UI. HTTP 503 is included here. |
| `GEMINI_DNS_ERROR` / `GEMINI_PROXY_ERROR` / `GEMINI_NETWORK_ERROR` | Deployment/network problem; alert backend operations. |
| `GEMINI_MODEL_UNAVAILABLE` | Configured model is unavailable to the project. |

Tool-result statuses:

| Status | Meaning |
|---|---|
| `OK` | A real adapter returned validated JSON data. |
| `MOCK` | Illustrative data only; never present it as real. |
| `SCENARIO_REQUIRED` | No trusted scenario ID was available. |
| `INVALID_ARGUMENT` / `UNKNOWN_TOOL` | The requested call failed allowlist/schema validation. |
| `SERVICE_UNAVAILABLE` | Adapter missing or downstream call failed. |
| `BACKEND_ERROR` | Adapter returned the wrong type or non-JSON data. |
| `CANCELLED` | User cancelled a pending simulation start. |

Do not return raw exceptions, API keys, internal URLs, credentials, or downstream response bodies to Gemini or the frontend.

## Environment variables

Required Member 1 variables:

```dotenv
GEMINI_API_KEY=<local-or-secret-manager-value>
GEMINI_MODEL=gemini-3.8-flash
AI_BACKEND_MODE=mock
```

- `GEMINI_API_KEY`: server-side only; inject through the deployment secret manager.
- `GEMINI_MODEL`: currently `gemini-3.8-flash`.
- `AI_BACKEND_MODE`: keep `mock` until both real adapters pass integration tests.

Member 2/3 service URLs, tokens, and timeouts should use teammate-owned environment names and be read inside their adapter implementations. Do not add their secrets to frontend configuration or tool results.

## Safe transition from mock to real

1. Implement and unit-test both adapter objects against their owned services.
2. Inject both through `BackendAdapters`; do not edit the tool allowlist or bypass validation.
3. In a non-production environment, instantiate `ToolRegistry(mode="real", adapters=...)` explicitly and test all read-only methods.
4. Verify returned values are JSON serializable and include the requested `scenario_id`.
5. Test simulation start with explicit confirmation and verify Member 3 idempotency.
6. Test downstream timeout/unavailable behavior and confirm the frontend receives safe statuses.
7. Confirm logs contain no user prompts, credentials, or private provider/downstream messages.
8. Change the deployed backend's `AI_BACKEND_MODE` from `mock` to `real` through deployment configuration. Do not edit committed source or `.env.example` for the rollout.
9. Restart the backend so the registry reads the new environment value.
10. Run one controlled smoke test per tool, monitor failures, and roll back to `mock` if adapters are incomplete or unsafe.

Do not enable real mode with either adapter absent. Real mode deliberately returns `SERVICE_UNAVAILABLE` rather than substituting mock results.

## Handoff checklist

### Member 2

- Implement the two exact `FloodBackendAdapter` methods.
- Return strict JSON mappings with scenario, horizon, freshness, and source-derived hazard data.
- Supply the initialized flood adapter to the backend composition owner.
- Add service-specific timeout, authentication, and failure tests.

### Member 3

- Implement the four exact `RescueBackendAdapter` methods.
- Make simulation start idempotent and preserve the explicit-confirmation boundary.
- Return strict JSON mappings with scenario IDs and backend-derived state/routes/events.
- Supply the initialized rescue adapter to the backend composition owner.

### Member 4

- Build chat UI against `POST /api/ai/ask` in the team backend.
- Persist `session_id`, support confirmation/cancellation, and render mock/real labels.
- Handle assistant statuses without exposing internal diagnostics.
- Never call Gemini or hold `GEMINI_API_KEY` in browser code.

### Backend integration owner

- Construct both teammate adapters and inject `BackendAdapters` into `ToolRegistry`.
- Keep one long-lived `AIAssistant` instance per process and address session affinity before scaling.
- Add authentication, request parsing, rate limiting, and the `/api/ai/ask` route.
- Keep `AI_BACKEND_MODE=mock` until the safe-transition checklist passes.

