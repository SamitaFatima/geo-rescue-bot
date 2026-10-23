"""
api.py - RescueBot 2.0 (Member 3: Routing & Backend)

The team's HTTP backend (Flask). It mounts the endpoints from the shared
data contract in the revised team report, section 4:

  GET  /api/geo/area              -> Member 2's area info
  POST /api/hazards/simulate      -> Member 2's time-stepped hazard snapshots
  POST /api/simulation/start      -> start (idempotent) a rescue simulation
  GET  /api/simulation/state      -> agents, victims, hazards, routes, metrics
  POST /api/simulation/step       -> advance to the next flood snapshot
  POST /api/simulation/replan     -> re-plan the route now, with the reason
  POST /api/ai/ask                -> Member 1's assistant (this file only wires it)
  GET  /api/health                -> liveness + backend mode

HOW TO RUN
----------
    pip install flask
    python api.py                  # http://127.0.0.1:5000

Environment variables (all optional):
    AI_BACKEND_MODE   mock (default in .env.example) or real - Member 1's switch
    API_HOST          default 127.0.0.1
    API_PORT          default 5000
    CORS_ORIGINS      comma-separated allowed browser origins for Member 4's
                      React dev server; default http://localhost:5173,
                      http://localhost:3000, http://127.0.0.1:5173

HTTP ERROR MAPPING (documented for Member 4)
--------------------------------------------
    400 malformed JSON / missing or invalid field
    404 unknown scenario_id, or unknown route
    409 simulation not started yet / already finished
    500 unexpected server error (generic message only - no internals leaked)
    503 the AI assistant could not be created (e.g. Gemini not configured)
  /api/ai/ask returns the assistant's own domain result with HTTP 200 and a
  `status` field (OK, CONFIRMATION_REQUIRED, ...) exactly as documented in
  MEMBER1_INTEGRATION_HANDOFF.md.

Every success body that carries scenario numbers includes the disclaimer
"Illustrative scenario estimates, not verified forecasts."
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory
from werkzeug.exceptions import HTTPException

import geo_data
from gis_service import FloodServiceAdapter, GISService
from simulation_service import RescueServiceAdapter, SimulationError, SimulationService

log = logging.getLogger("rescuebot.api")

_SCENARIO_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_HTTP_BY_CODE = {
    "UNKNOWN_SCENARIO": 404,
    "NOT_STARTED": 409,
    "SIMULATION_FINISHED": 409,
}
MAX_BODY_BYTES = 64 * 1024


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _json_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError(400, "INVALID_JSON", "Request body must be a JSON object.")
    return data


def _scenario_id(value: Any) -> str:
    if not isinstance(value, str) or not _SCENARIO_RE.fullmatch(value):
        raise ApiError(400, "INVALID_SCENARIO_ID",
                       "scenario_id must be 1-64 characters: letters, digits, '_' or '-'.")
    return value


def create_app(
    *,
    assistant: Any | None = None,
    simulation_service: SimulationService | None = None,
    gis: GISService | None = None,
) -> Flask:
    """Application factory. Tests pass fakes; production uses the defaults.

    `assistant` is Member 1's AIAssistant. When omitted it is built lazily
    on the first /api/ai/ask call (so the rest of the API works even if the
    Gemini configuration is missing), wired to the SAME simulation service
    the HTTP endpoints use - the assistant and the dashboard always see the
    same simulation.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    gis = gis or GISService()
    service = simulation_service or SimulationService(gis)
    holder: dict[str, Any] = {"assistant": assistant}
    build_lock = threading.Lock()

    allowed_origins = {
        o.strip() for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5500,http://localhost:5500",
        ).split(",") if o.strip()
    }

    @app.after_request
    def _cors(response):
        origin = request.headers.get("Origin")
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.before_request
    def _preflight():
        if request.method == "OPTIONS":
            return ("", 204)
        return None

    # ---------------- error handling ----------------
    def _error(status: int, code: str, message: str):
        return jsonify({"status": "ERROR", "error": {"code": code, "message": message}}), status

    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.errorhandler(SimulationError)
    def _sim_error(exc: SimulationError):
        return _error(_HTTP_BY_CODE.get(exc.code, 400), exc.code, exc.message)

    @app.errorhandler(geo_data.UnknownScenarioError)
    def _unknown(exc):
        return _error(404, "UNKNOWN_SCENARIO", "scenario_id is not a prepared scenario.")

    @app.errorhandler(404)
    def _not_found(_exc):
        return _error(404, "NOT_FOUND", "No such endpoint.")

    @app.errorhandler(405)
    def _bad_method(_exc):
        return _error(405, "METHOD_NOT_ALLOWED", "Method not allowed for this endpoint.")

    @app.errorhandler(413)
    def _too_big(_exc):
        return _error(400, "BODY_TOO_LARGE", "Request body is too large.")

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        if isinstance(exc, HTTPException):
            return _error(exc.code or 500, "HTTP_ERROR", exc.name)
        # Log the type only: messages can contain prompts or internal details.
        log.error("Unexpected error: %s", type(exc).__name__)
        return _error(500, "INTERNAL_ERROR", "Unexpected server error.")

    # ---------------- GUI (single-page dashboard in ./static) ----------------
    @app.get("/")
    def dashboard():
        return redirect("/static/index.html")

    # ---------------- Member 2 endpoints (mounted) ----------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "OK",
                        "backend_mode": os.getenv("AI_BACKEND_MODE", "unset")})

    @app.get("/api/geo/area")
    def geo_area():
        return jsonify(gis.get_area_info())

    @app.post("/api/hazards/simulate")
    def hazards_simulate():
        body = _json_body()
        scenario_id = _scenario_id(body.get("scenario_id"))
        steps = body.get("time_steps")
        if steps is None:
            return jsonify(gis.get_hazard_snapshot(scenario_id))
        if (not isinstance(steps, list) or not steps or len(steps) > 24
                or any(isinstance(h, bool) or not isinstance(h, int) or not 0 <= h <= 168
                       for h in steps)):
            raise ApiError(400, "INVALID_TIME_STEPS",
                           "time_steps must be a list of 1-24 integers from 0 to 168.")
        return jsonify(gis.get_hazard_snapshot(scenario_id, tuple(steps)))

    # ---------------- Member 3 endpoints ----------------
    @app.post("/api/simulation/start")
    def simulation_start():
        body = _json_body()
        scenario_id = _scenario_id(body.get("scenario_id"))
        restart = body.get("restart", False)
        if not isinstance(restart, bool):
            raise ApiError(400, "INVALID_RESTART", "restart must be a boolean.")
        return jsonify(service.start(scenario_id, restart=restart))

    @app.get("/api/simulation/state")
    def simulation_state():
        return jsonify(service.state(_scenario_id(request.args.get("scenario_id"))))

    @app.post("/api/simulation/step")
    def simulation_step():
        body = _json_body()
        return jsonify(service.step(_scenario_id(body.get("scenario_id"))))

    @app.post("/api/simulation/replan")
    def simulation_replan():
        body = _json_body()
        scenario_id = _scenario_id(body.get("scenario_id"))
        reason = body.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 200):
            raise ApiError(400, "INVALID_REASON", "reason must be text up to 200 characters.")
        return jsonify(service.replan(scenario_id, reason))

    # ---------------- Member 1 endpoint (wired, not implemented, here) ----------------
    def _get_assistant():
        with build_lock:
            return _build_assistant()

    def _build_assistant():
        if holder["assistant"] is None:
            # Imported lazily: the rest of the API must not depend on the
            # Gemini SDK or an API key being present.
            from ai_agent import AIAssistant
            from ai_tools import BackendAdapters, create_tool_registry_from_env
            from gemini_service import GeminiService

            adapters = BackendAdapters(
                flood=FloodServiceAdapter(gis),
                rescue=RescueServiceAdapter(service),
            )
            holder["assistant"] = AIAssistant(
                gemini_service=GeminiService(),
                tool_registry=create_tool_registry_from_env(adapters=adapters),
            )
        return holder["assistant"]

    @app.post("/api/ai/ask")
    def ai_ask():
        body = _json_body()
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ApiError(400, "INVALID_MESSAGE", "message must be non-empty text.")
        session_id = body.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise ApiError(400, "INVALID_SESSION_ID", "session_id must be text.")
        confirm = body.get("confirm", False)
        if not isinstance(confirm, bool):
            raise ApiError(400, "INVALID_CONFIRM", "confirm must be a boolean.")
        try:
            assistant_ = _get_assistant()
        except Exception as exc:  # configuration / SDK problems
            log.error("Assistant unavailable: %s", type(exc).__name__)
            return _error(503, "ASSISTANT_UNAVAILABLE",
                          "The AI assistant is not available right now.")
        return jsonify(assistant_.ask(message, session_id=session_id, confirm=confirm))

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=os.getenv("API_HOST", "127.0.0.1"),
                     port=int(os.getenv("API_PORT", "5000")))
