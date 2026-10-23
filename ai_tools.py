"""Allowlisted, validated interfaces to Member 2 and Member 3 backends.

No GIS, flood calculations, routing, or rescue simulation are implemented here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

TOOL_NAMES = (
    "get_flood_risk",
    "get_affected_areas",
    "start_rescue_simulation",
    "get_rescue_routes",
    "get_simulation_status",
    "explain_route_change",
)

_SCENARIO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SCENARIO_SCHEMA = {
    "type": "string",
    "description": "Identifier for an existing backend scenario.",
    "minLength": 1,
    "maxLength": 64,
    "pattern": _SCENARIO_PATTERN.pattern,
}
_HOURS_SCHEMA = {
    "type": "integer",
    "description": "Forecast horizon in hours, from 1 through 168.",
    "minimum": 1,
    "maximum": 168,
}

TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_flood_risk",
        "description": "Get flood-risk results from the connected flood backend.",
        "parameters": {
            "type": "object",
            "properties": {"hours": _HOURS_SCHEMA, "scenario_id": _SCENARIO_SCHEMA},
            "required": ["hours"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_affected_areas",
        "description": "Get affected-area results from the connected flood backend.",
        "parameters": {
            "type": "object",
            "properties": {"scenario_id": _SCENARIO_SCHEMA, "hours": _HOURS_SCHEMA},
            "required": ["hours"],
            "additionalProperties": False,
        },
    },
    {
        "name": "start_rescue_simulation",
        "description": "Start a rescue simulation after explicit user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {"scenario_id": _SCENARIO_SCHEMA},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_rescue_routes",
        "description": "Get rescue routes from the connected rescue backend.",
        "parameters": {
            "type": "object",
            "properties": {"scenario_id": _SCENARIO_SCHEMA},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_simulation_status",
        "description": "Get simulation status and reported rescue counts.",
        "parameters": {
            "type": "object",
            "properties": {"scenario_id": _SCENARIO_SCHEMA},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_route_change",
        "description": "Explain a route change using backend-provided event data.",
        "parameters": {
            "type": "object",
            "properties": {"scenario_id": _SCENARIO_SCHEMA},
            "required": [],
            "additionalProperties": False,
        },
    },
)


class FloodBackendAdapter(Protocol):
    """Interface implemented by Member 2's flood-analysis adapter."""

    def get_flood_risk(self, hours: int, scenario_id: str) -> Mapping[str, Any]: ...

    def get_affected_areas(
        self, scenario_id: str, hours: int
    ) -> Mapping[str, Any]: ...


class RescueBackendAdapter(Protocol):
    """Interface implemented by Member 3's rescue-simulation adapter."""

    def start_rescue_simulation(self, scenario_id: str) -> Mapping[str, Any]: ...

    def get_rescue_routes(self, scenario_id: str) -> Mapping[str, Any]: ...

    def get_simulation_status(self, scenario_id: str) -> Mapping[str, Any]: ...

    def explain_route_change(self, scenario_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BackendAdapters:
    """Optional backend adapters; absent adapters never trigger mock fallback."""

    flood: FloodBackendAdapter | None = None
    rescue: RescueBackendAdapter | None = None


class ToolValidationError(ValueError):
    """Raised for invalid tool names or arguments."""


def _validate_scenario(value: Any) -> str:
    if not isinstance(value, str) or not _SCENARIO_PATTERN.fullmatch(value):
        raise ToolValidationError(
            "scenario_id must be 1-64 letters, digits, underscores, or hyphens."
        )
    return value


def _validate_hours(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 168:
        raise ToolValidationError("hours must be an integer from 1 through 168.")
    return value


def _validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        raise ToolValidationError("Unknown tool.")
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("Tool arguments must be an object.")

    expected: dict[str, set[str]] = {
        "get_flood_risk": {"hours", "scenario_id"},
        "get_affected_areas": {"scenario_id", "hours"},
        "start_rescue_simulation": {"scenario_id"},
        "get_rescue_routes": {"scenario_id"},
        "get_simulation_status": {"scenario_id"},
        "explain_route_change": {"scenario_id"},
    }
    allowed = expected[name]
    required = allowed - {"scenario_id"}
    if not required.issubset(arguments) or not set(arguments).issubset(allowed):
        raise ToolValidationError("Tool arguments do not match the declared schema.")

    validated = {}
    if "scenario_id" in arguments:
        validated["scenario_id"] = _validate_scenario(arguments["scenario_id"])
    if name in ("get_flood_risk", "get_affected_areas"):
        validated["hours"] = _validate_hours(arguments["hours"])
    return validated


class ToolRegistry:
    """Dispatches only registered tools to explicit mock or real backends."""

    def __init__(
        self,
        *,
        mode: str | None = None,
        adapters: BackendAdapters | None = None,
        default_scenario_id: str | None = None,
        mock_fixtures: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]]
        | None = None,
    ) -> None:
        load_dotenv(override=False)
        selected_mode = (
            mode if mode is not None else os.getenv("AI_BACKEND_MODE", "real")
        ).strip().lower()
        if selected_mode not in ("mock", "real"):
            raise ValueError("AI_BACKEND_MODE must be 'mock' or 'real'.")
        self.mode = selected_mode
        self.adapters = adapters or BackendAdapters()
        self.default_scenario_id = (
            _validate_scenario(default_scenario_id)
            if default_scenario_id is not None
            else None
        )
        self._mock_fixtures = dict(
            MOCK_FIXTURES if mock_fixtures is None else mock_fixtures
        )

    @property
    def function_declarations(self) -> tuple[dict[str, Any], ...]:
        return TOOL_DECLARATIONS

    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        """Validate a model-requested call without executing it."""
        if (
            isinstance(arguments, Mapping)
            and "scenario_id" not in arguments
            and self.default_scenario_id is not None
        ):
            arguments = {**arguments, "scenario_id": self.default_scenario_id}
        if (
            isinstance(arguments, Mapping)
            and name in TOOL_NAMES
            and "scenario_id" not in arguments
        ):
            raise ToolValidationError("scenario_id is required.")
        return _validate_arguments(name, arguments)

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        """Validate, dispatch, and normalize one allowlisted tool call."""
        if name not in TOOL_NAMES:
            return self._result("UNKNOWN_TOOL", name, {}, "Unknown tool.")
        if (
            isinstance(arguments, Mapping)
            and "scenario_id" not in arguments
            and self.default_scenario_id is None
        ):
            try:
                _validate_arguments(
                    name, {**arguments, "scenario_id": "validation-placeholder"}
                )
            except ToolValidationError as error:
                return self._result(
                    "INVALID_ARGUMENT", name, {}, str(error)
                )
            return self._result(
                "SCENARIO_REQUIRED",
                name,
                {},
                "No scenario ID was provided. Ask the user which scenario to use.",
            )
        try:
            validated = self.validate(name, arguments)
        except ToolValidationError as error:
            status = "UNKNOWN_TOOL" if name not in TOOL_NAMES else "INVALID_ARGUMENT"
            return self._result(status, name, {}, str(error))

        if self.mode == "mock":
            fixture = self._mock_fixtures.get(name, _mock_fixture)
            try:
                data = fixture(validated)
            except Exception as error:
                logger.warning("Mock fixture for %s failed (%s).", name, type(error).__name__)
                return self._result(
                    "BACKEND_ERROR", name, {}, "The mock fixture failed."
                )
            if not isinstance(data, Mapping):
                return self._result(
                    "BACKEND_ERROR", name, {}, "The mock fixture returned invalid data."
                )
            if not _is_json_serializable(data):
                return self._result(
                    "BACKEND_ERROR",
                    name,
                    {},
                    "The mock fixture returned data that cannot be serialized.",
                )
            return self._result(
                "MOCK",
                name,
                dict(data),
                "Illustrative fixture only; this is not real backend data.",
            )

        adapter, method_name = _real_adapter(self.adapters, name)
        if adapter is None:
            return self._result(
                "SERVICE_UNAVAILABLE",
                name,
                {},
                f"No real backend adapter is connected for {name}.",
            )

        try:
            data = getattr(adapter, method_name)(**validated)
        except Exception as error:
            logger.warning(
                "Backend call for %s failed (%s).", name, type(error).__name__
            )
            return self._result(
                "SERVICE_UNAVAILABLE",
                name,
                {},
                "The connected backend could not complete this request.",
            )
        if not isinstance(data, Mapping):
            logger.warning("Backend call for %s returned an invalid result type.", name)
            return self._result(
                "BACKEND_ERROR", name, {}, "The backend returned an invalid result."
            )
        if not _is_json_serializable(data):
            logger.warning("Backend call for %s returned non-JSON data.", name)
            return self._result(
                "BACKEND_ERROR",
                name,
                {},
                "The backend returned data that cannot be serialized.",
            )
        return self._result("OK", name, dict(data), "Result returned by the real backend.")

    def _result(
        self, status: str, tool: str, data: dict[str, Any], message: str
    ) -> dict[str, Any]:
        return {
            "status": status,
            "mode": self.mode,
            "label": "MOCK" if self.mode == "mock" else "REAL",
            "tool": tool,
            "data": data,
            "message": message,
        }


def _real_adapter(
    adapters: BackendAdapters, name: str
) -> tuple[Any | None, str]:
    if name in ("get_flood_risk", "get_affected_areas"):
        return adapters.flood, name
    return adapters.rescue, name


def _is_json_serializable(data: Mapping[str, Any]) -> bool:
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _mock_fixture(arguments: dict[str, Any]) -> Mapping[str, Any]:
    name = arguments.get("_tool_name")
    if name == "get_flood_risk":
        return {
            "scenario_id": arguments["scenario_id"],
            "hours": arguments["hours"],
            "risk_assessment": "NOT_ASSESSED",
            "detail": "No flood calculation or forecast is provided in mock mode.",
        }
    if name == "get_affected_areas":
        return {
            "scenario_id": arguments["scenario_id"],
            "hours": arguments["hours"],
            "areas": [],
            "detail": "No geographic analysis is provided in mock mode.",
        }
    if name == "start_rescue_simulation":
        return {
            "scenario_id": arguments["scenario_id"],
            "started": False,
            "detail": "No rescue simulation is run in mock mode.",
        }
    if name == "get_rescue_routes":
        return {
            "scenario_id": arguments["scenario_id"],
            "routes": [],
            "detail": "No routes are computed in mock mode.",
        }
    if name == "get_simulation_status":
        return {
            "scenario_id": arguments["scenario_id"],
            "status": "NOT_AVAILABLE",
            "victims_rescued": None,
            "detail": "No simulation or rescue counts are available in mock mode.",
        }
    if name == "explain_route_change":
        return {
            "scenario_id": arguments["scenario_id"],
            "explanation": None,
            "detail": "No route-change events are available in mock mode.",
        }
    raise ToolValidationError("Unknown tool.")


MOCK_FIXTURES: dict[str, Callable[[dict[str, Any]], Mapping[str, Any]]] = {
    name: (lambda args, tool=name: _mock_fixture({**args, "_tool_name": tool}))
    for name in TOOL_NAMES
}


def create_tool_registry_from_env(
    *, adapters: BackendAdapters | None = None
) -> ToolRegistry:
    """Build a registry using the explicit ``AI_BACKEND_MODE`` setting."""
    return ToolRegistry(adapters=adapters, mock_fixtures=MOCK_FIXTURES)
