from __future__ import annotations

import logging
import socket
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors

from ai_agent import AIAssistant
from ai_tools import BackendAdapters, TOOL_NAMES, ToolRegistry
import gemini_service
from gemini_service import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiRateLimitError,
    GeminiRequestError,
    GeminiService,
    GeminiTimeoutError,
    GeminiUnavailableError,
)


def text_response(text: str) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))
        ]
    )


def function_response(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name=name, args=arguments, id=call_id
                            )
                        )
                    ]
                )
            )
        ]
    )


class FakeGeminiClient:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.models = self

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Gemini request.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAPIException(Exception):
    def __init__(self, code: int):
        super().__init__("simulated HTTP failure")
        self.code = code


class FakeFloodBackend:
    def get_flood_risk(self, hours: int, scenario_id: str) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "hours": hours, "risk": "backend-reported"}

    def get_affected_areas(self, scenario_id: str, hours: int) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "hours": hours, "areas": ["backend-area"]}


class FakeRescueBackend:
    def start_rescue_simulation(self, scenario_id: str) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "started": True}

    def get_rescue_routes(self, scenario_id: str) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "routes": ["backend-route"]}

    def get_simulation_status(self, scenario_id: str) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "victims_rescued": 4}

    def explain_route_change(self, scenario_id: str) -> dict[str, Any]:
        return {"scenario_id": scenario_id, "reason": "backend-event"}


def make_assistant(
    responses: list[Any],
    *,
    registry: ToolRegistry | None = None,
    max_tool_rounds: int = 5,
) -> tuple[AIAssistant, FakeGeminiClient]:
    client = FakeGeminiClient(responses)
    service = GeminiService(client=client, max_retries=0)
    assistant = AIAssistant(
        gemini_service=service,
        tool_registry=registry
        or ToolRegistry(mode="mock", default_scenario_id="scenario-1"),
        max_tool_rounds=max_tool_rounds,
    )
    return assistant, client


def test_gemini_service_sends_configured_model_and_function_declarations() -> None:
    client = FakeGeminiClient([text_response("Ready.")])
    service = GeminiService(
        client=client, model="gemini-test-model", max_retries=0
    )
    declaration = {
        "name": "get_simulation_status",
        "description": "Read simulation status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    result = service.generate_content(
        contents="hello",
        function_declarations=[declaration],
        system_instruction="Be safe.",
    )

    assert result.candidates
    assert client.calls[0]["model"] == "gemini-test-model"
    config = client.calls[0]["config"]
    assert config.system_instruction == "Be safe."
    assert config.tools[0].function_declarations[0].name == "get_simulation_status"
    assert config.automatic_function_calling.disable is True


def test_missing_gemini_credentials_are_reported_without_network(
    monkeypatch: Any
) -> None:
    monkeypatch.setattr(gemini_service, "load_dotenv", lambda **_: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = GeminiService(max_retries=0)

    with pytest.raises(GeminiConfigurationError):
        service.generate_content(contents="hello", system_instruction="safe")


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_flood_risk", {"hours": 10, "scenario_id": "scenario-1"}),
        ("get_affected_areas", {"scenario_id": "scenario-1", "hours": 10}),
        ("start_rescue_simulation", {"scenario_id": "scenario-1"}),
        ("get_rescue_routes", {"scenario_id": "scenario-1"}),
        ("get_simulation_status", {"scenario_id": "scenario-1"}),
        ("explain_route_change", {"scenario_id": "scenario-1"}),
    ],
)
def test_all_six_tools_return_explicit_mock_fixtures(
    tool_name: str, arguments: dict[str, Any]
) -> None:
    result = ToolRegistry(mode="mock").execute(tool_name, arguments)
    assert result["status"] == "MOCK"
    assert result["label"] == "MOCK"
    assert result["mode"] == "mock"
    assert result["tool"] == tool_name
    assert "Illustrative fixture only" in result["message"]
    assert result["data"]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_flood_risk", {"hours": True, "scenario_id": "s-1"}),
        ("get_flood_risk", {"hours": 0, "scenario_id": "s-1"}),
        ("get_flood_risk", {"hours": 169, "scenario_id": "s-1"}),
        ("get_affected_areas", {"scenario_id": "bad scenario", "hours": 1}),
        ("get_rescue_routes", {"scenario_id": "s-1", "extra": 1}),
    ],
)
def test_invalid_tool_arguments_are_rejected(
    tool_name: str, arguments: dict[str, Any]
) -> None:
    result = ToolRegistry(mode="mock").execute(tool_name, arguments)
    assert result["status"] == "INVALID_ARGUMENT"


def test_missing_scenario_id_is_requested_not_treated_as_backend_data() -> None:
    result = ToolRegistry(mode="mock").execute("get_simulation_status", {})
    assert result["status"] == "SCENARIO_REQUIRED"


def test_unknown_tools_are_rejected() -> None:
    result = ToolRegistry(mode="mock").execute("delete_database", {})
    assert result["status"] == "UNKNOWN_TOOL"


def test_real_mode_delegates_without_mock_fallback() -> None:
    registry = ToolRegistry(
        mode="real",
        adapters=BackendAdapters(flood=FakeFloodBackend(), rescue=FakeRescueBackend()),
    )
    result = registry.execute("get_flood_risk", {"hours": 5, "scenario_id": "s-1"})
    assert result["status"] == "OK"
    assert result["data"]["risk"] == "backend-reported"
    assert result["label"] == "REAL"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_flood_risk", {"hours": 8, "scenario_id": "s-1"}),
        ("get_affected_areas", {"scenario_id": "s-1", "hours": 8}),
        ("start_rescue_simulation", {"scenario_id": "s-1"}),
        ("get_rescue_routes", {"scenario_id": "s-1"}),
        ("get_simulation_status", {"scenario_id": "s-1"}),
        ("explain_route_change", {"scenario_id": "s-1"}),
    ],
)
def test_all_tools_dispatch_to_real_adapters(
    name: str, arguments: dict[str, Any]
) -> None:
    registry = ToolRegistry(
        mode="real",
        adapters=BackendAdapters(flood=FakeFloodBackend(), rescue=FakeRescueBackend()),
    )
    result = registry.execute(name, arguments)
    assert result["status"] == "OK"
    assert result["tool"] == name
    assert result["label"] == "REAL"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_flood_risk", {"hours": 10, "scenario_id": "scenario-1"}),
        ("get_affected_areas", {"scenario_id": "scenario-1", "hours": 10}),
        ("start_rescue_simulation", {"scenario_id": "scenario-1"}),
        ("get_rescue_routes", {"scenario_id": "scenario-1"}),
        ("get_simulation_status", {"scenario_id": "scenario-1"}),
        ("explain_route_change", {"scenario_id": "scenario-1"}),
    ],
)
def test_gemini_tool_calls_dispatch_all_allowlisted_tools(
    name: str, arguments: dict[str, Any]
) -> None:
    responses = [function_response(name, arguments)]
    if name == "start_rescue_simulation":
        responses.append(text_response("The mock tool did not run a simulation."))
    else:
        responses.append(text_response("Tool result received."))
    assistant, _ = make_assistant(responses)

    result = assistant.ask(f"Use {name}.")
    if name == "start_rescue_simulation":
        assert result["status"] == "CONFIRMATION_REQUIRED"
        result = assistant.ask("confirm", session_id=result["session_id"])
    assert name in result["tools_executed"]
    assert result["tool_results"][0]["tool"] == name


def test_missing_scenario_is_clarified_not_guessed() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "get_flood_risk",
                {"hours": 10, "scenario_id": "model-guessed-scenario"},
            ),
            text_response("An arbitrary scenario should not be selected."),
            function_response(
                "get_flood_risk",
                {"hours": 10, "scenario_id": "user-selected-scenario"},
                call_id="call-2",
            ),
            text_response("The tool result was received."),
        ],
        registry=ToolRegistry(mode="mock"),
    )
    clarification = assistant.ask("Show flood risk after 10 hours.")
    assert clarification["status"] == "OK"
    assert clarification["reply"] == "Which scenario ID should I use?"
    assert clarification["tool_results"][0]["status"] == "SCENARIO_REQUIRED"

    resolved = assistant.ask(
        "user-selected-scenario", session_id=clarification["session_id"]
    )
    assert resolved["tool_results"][0]["data"]["scenario_id"] == "user-selected-scenario"
    assert resolved["tool_results"][0]["label"] == "MOCK"


def test_model_cannot_override_the_user_selected_scenario() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "get_flood_risk",
                {"hours": 2, "scenario_id": "model-guessed-scenario"},
            ),
            text_response("A result for the wrong scenario should not be shown."),
        ],
        registry=ToolRegistry(mode="mock"),
    )
    result = assistant.ask(
        "Check flood risk for scenario_id: user-scenario.",
    )
    assert result["reply"] == "Which scenario ID should I use?"
    assert result["tool_results"][0]["status"] == "SCENARIO_REQUIRED"
    assert result["tools_executed"] == []


def test_trusted_default_scenario_id_is_injected_explicitly() -> None:
    registry = ToolRegistry(mode="mock", default_scenario_id="active-scenario")
    result = registry.execute("get_rescue_routes", {})
    assert result["status"] == "MOCK"
    assert result["data"]["scenario_id"] == "active-scenario"


def test_real_mode_without_adapters_is_unavailable_not_mock() -> None:
    result = ToolRegistry(mode="real").execute(
        "get_flood_risk", {"hours": 5, "scenario_id": "s-1"}
    )
    assert result["status"] == "SERVICE_UNAVAILABLE"
    assert result["label"] == "REAL"
    assert result["data"] == {}


def test_real_backend_exception_is_sanitized_and_unavailable(caplog: Any) -> None:
    class FailingBackend:
        def get_flood_risk(self, hours: int, scenario_id: str) -> dict[str, Any]:
            raise RuntimeError("secret-token private-payload")

    with caplog.at_level(logging.WARNING):
        result = ToolRegistry(
            mode="real", adapters=BackendAdapters(flood=FailingBackend())
        ).execute("get_flood_risk", {"hours": 2, "scenario_id": "s-1"})
    assert result["status"] == "SERVICE_UNAVAILABLE"
    assert "secret-token" not in caplog.text
    assert "private-payload" not in caplog.text


def test_backend_non_json_values_are_rejected() -> None:
    class InvalidBackend:
        def get_simulation_status(self, scenario_id: str) -> dict[str, Any]:
            return {"scenario_id": scenario_id, "value": object()}

    result = ToolRegistry(
        mode="real", adapters=BackendAdapters(rescue=InvalidBackend())
    ).execute("get_simulation_status", {"scenario_id": "s-1"})
    assert result["status"] == "BACKEND_ERROR"
    assert result["data"] == {}


def test_mock_fixtures_never_claim_real_counts_routes_or_forecasts() -> None:
    registry = ToolRegistry(mode="mock")
    status = registry.execute("get_simulation_status", {"scenario_id": "s-1"})
    routes = registry.execute("get_rescue_routes", {"scenario_id": "s-1"})
    risk = registry.execute(
        "get_flood_risk", {"hours": 5, "scenario_id": "s-1"}
    )
    assert status["data"]["victims_rescued"] is None
    assert routes["data"]["routes"] == []
    assert risk["data"]["risk_assessment"] == "NOT_ASSESSED"
    assert "not real backend data" in risk["message"]


def test_natural_language_tool_call_is_dispatched_and_context_is_sent() -> None:
    assistant, client = make_assistant(
        [
            function_response(
                "get_flood_risk", {"hours": 10, "scenario_id": "scenario-1"}
            ),
            text_response("Mock details are not available."),
            function_response(
                "get_flood_risk", {"hours": 5, "scenario_id": "scenario-1"},
                call_id="call-2",
            ),
            text_response("A five-hour update has been checked."),
        ]
    )
    first = assistant.ask("Show flood risk after 10 hours.", session_id=None)
    followup = assistant.ask(
        "What about after 5 hours?", session_id=first["session_id"]
    )
    assert first["tools_executed"] == ["get_flood_risk"]
    assert first["backend_mode"] == "mock"
    assert "MOCK" in followup["reply"]
    assert followup["tool_results"][0]["data"]["hours"] == 5
    assert len(client.calls) == 4
    assert len(client.calls[-1]["contents"]) >= 3


def test_sessions_are_isolated() -> None:
    assistant, client = make_assistant([text_response("One."), text_response("Two.")])
    first = assistant.ask("First session")
    second = assistant.ask("Second session")
    assert first["session_id"] != second["session_id"]
    first_history = client.calls[0]["contents"]
    second_history = client.calls[1]["contents"]
    assert "First session" not in repr(second_history)
    assert "Second session" not in repr(first_history)
    assert assistant.ask("Bad session", session_id="unknown")["status"] == "SESSION_NOT_FOUND"


def test_conversation_history_is_bounded() -> None:
    client = FakeGeminiClient(
        [text_response("First."), text_response("Second."), text_response("Third.")]
    )
    assistant = AIAssistant(
        gemini_service=GeminiService(client=client, max_retries=0),
        tool_registry=ToolRegistry(mode="mock"),
        max_history_items=4,
    )
    first = assistant.ask("First turn.")
    assistant.ask("Second turn.", session_id=first["session_id"])
    assistant.ask("Third turn.", session_id=first["session_id"])
    assert len(assistant._sessions[first["session_id"]].history) <= 4


def test_invalid_session_and_confirmation_types_are_rejected() -> None:
    assistant, _ = make_assistant([text_response("unused")])
    assert assistant.ask("Hello", session_id=[])["status"] == "INVALID_REQUEST"
    assert assistant.ask("Hello", confirm="yes")["status"] == "INVALID_REQUEST"


def test_simulation_requires_confirmation_before_tool_execution() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "start_rescue_simulation", {"scenario_id": "scenario-1"}
            ),
            text_response("The mock mode did not start a simulation."),
        ]
    )
    first = assistant.ask("Start the rescue simulation.")
    assert first["status"] == "CONFIRMATION_REQUIRED"
    assert first["needs_confirmation"] is True

    reminder = assistant.ask("Maybe later.", session_id=first["session_id"])
    assert reminder["status"] == "CONFIRMATION_REQUIRED"

    confirmed = assistant.ask("confirm", session_id=first["session_id"])
    assert confirmed["status"] == "OK"


def test_simulation_can_be_cancelled() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "start_rescue_simulation", {"scenario_id": "scenario-1"}
            ),
            text_response("The simulation request was cancelled."),
        ]
    )
    pending = assistant.ask("Start the rescue simulation.")
    result = assistant.ask("cancel", session_id=pending["session_id"])
    assert result["status"] == "OK"
    assert "cancelled" in result["reply"]


def test_explicit_confirm_flag_confirms_pending_action() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "start_rescue_simulation", {"scenario_id": "scenario-1"}
            ),
            text_response("The mock tool confirms no simulation was run."),
        ]
    )
    pending = assistant.ask("Start the rescue simulation.")
    result = assistant.ask(
        "Proceed.", session_id=pending["session_id"], confirm=True
    )
    assert result["status"] == "OK"


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (GeminiAuthenticationError("private"), "AUTHENTICATION_FAILED"),
        (GeminiRateLimitError("private"), "QUOTA_EXHAUSTED"),
        (GeminiTimeoutError("private"), "GEMINI_TIMEOUT"),
        (GeminiUnavailableError("private"), "GEMINI_UNAVAILABLE"),
    ],
)
def test_assistant_reports_gemini_errors_safely(
    exception: Exception, expected: str, caplog: Any
) -> None:
    assistant, _ = make_assistant([exception])
    with caplog.at_level(logging.WARNING):
        result = assistant.ask("Hello")
    assert result["status"] == expected
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    ("error", "expected_category", "expected_status", "expected_class"),
    [
        (
            genai_errors.APIError(
                401,
                {
                    "error": {
                        "message": "secret provider text",
                        "status": "UNAUTHENTICATED",
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                "reason": "API_KEY_INVALID",
                                "domain": "googleapis.com",
                            }
                        ],
                    }
                },
            ),
            "AUTHENTICATION",
            401,
            GeminiAuthenticationError,
        ),
        (
            genai_errors.APIError(403, {"error": {"message": "secret provider text"}}),
            "ACCESS_DENIED",
            403,
            GeminiAuthenticationError,
        ),
        (
            genai_errors.APIError(429, {"error": {"message": "secret provider text"}}),
            "QUOTA_OR_RATE_LIMIT",
            429,
            GeminiRateLimitError,
        ),
        (
            genai_errors.APIError(503, {"error": {"message": "secret provider text"}}),
            "SERVICE_UNAVAILABLE",
            503,
            GeminiUnavailableError,
        ),
        (
            genai_errors.APIError(404, {"error": {"message": "secret provider text"}}),
            "MODEL_NOT_FOUND",
            404,
            Exception,
        ),
    ],
)
def test_google_sdk_api_errors_keep_sanitized_category_and_status(
    error: Exception,
    expected_category: str,
    expected_status: int,
    expected_class: type[Exception],
) -> None:
    assistant, _ = make_assistant([error])
    result = assistant.ask("Check the Gemini connection.")
    assert isinstance(error, genai_errors.APIError)
    translated = gemini_service._translate_error(error)
    assert isinstance(translated, expected_class)
    assert translated.diagnostic_category == expected_category
    assert translated.http_status == expected_status
    assert result["diagnostic_category"] == expected_category
    assert result["http_status"] == expected_status
    if expected_status == 401:
        assert result["diagnostic_reason"] == "API_KEY_INVALID"
    assert "secret provider text" not in repr(result)


def test_google_sdk_error_logs_only_category_and_status(caplog: Any) -> None:
    provider_message = "private provider text with sensitive detail"
    client = FakeGeminiClient(
        [
            genai_errors.APIError(
                503, {"error": {"message": provider_message}}
            )
        ]
    )
    service = GeminiService(client=client, max_retries=0)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(GeminiUnavailableError) as caught:
            service.generate_content(contents="hello", system_instruction="safe")
    assert caught.value.diagnostic_category == "SERVICE_UNAVAILABLE"
    assert caught.value.http_status == 503
    assert "category=SERVICE_UNAVAILABLE" in caplog.text
    assert "http_status=503" in caplog.text
    assert provider_message not in caplog.text


@pytest.mark.parametrize(
    ("reason", "expected_category"),
    [
        ("MODEL_NOT_FOUND", "MODEL_NOT_FOUND"),
        ("SERVICE_DISABLED", "API_NOT_ENABLED"),
        ("API_KEY_SERVICE_BLOCKED", "API_KEY_RESTRICTED"),
    ],
)
def test_google_structured_error_reason_is_safely_exposed(
    reason: str, expected_category: str, caplog: Any
) -> None:
    error = genai_errors.APIError(
        404,
        {
            "error": {
                "message": "private provider message",
                "status": "NOT_FOUND",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": reason,
                        "domain": "googleapis.com",
                        "metadata": {"private": "do not expose"},
                    }
                ],
            }
        },
    )
    client = FakeGeminiClient([error])
    service = GeminiService(client=client, max_retries=0)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(GeminiRequestError) as caught:
            service.generate_content(contents="hello", system_instruction="safe")

    assert caught.value.diagnostic_reason == reason
    assert caught.value.diagnostic_category == expected_category
    assert caught.value.http_status == 404
    assert reason in caplog.text
    assert "private provider message" not in caplog.text
    assert "do not expose" not in caplog.text


def test_google_error_reason_rejects_unstructured_sensitive_text() -> None:
    error = genai_errors.APIError(
        404,
        {
            "error": {
                "message": "sensitive message",
                "status": "NOT_FOUND",
                "details": [
                    {
                        "reason": "model missing for user@example.com",
                    }
                ],
            }
        },
    )

    translated = gemini_service._translate_error(error)

    assert translated.diagnostic_reason == "NOT_FOUND"
    assert "user@example.com" not in repr(translated)


@pytest.mark.parametrize(
    ("error", "expected_category", "expected_class"),
    [
        (httpx.TimeoutException("secret timeout detail"), "TIMEOUT", GeminiTimeoutError),
        (httpx.ProxyError("secret proxy detail"), "PROXY", GeminiUnavailableError),
        (httpx.ConnectError("secret connection detail"), "NETWORK", GeminiUnavailableError),
    ],
)
def test_httpx_connection_failures_are_safely_classified(
    error: Exception,
    expected_category: str,
    expected_class: type[Exception],
) -> None:
    translated = gemini_service._translate_error(error)
    assert isinstance(translated, expected_class)
    assert translated.diagnostic_category == expected_category
    assert translated.http_status is None
    assert "secret" not in str(translated)


def test_dns_cause_is_classified_without_raw_error_text() -> None:
    error = httpx.ConnectError("private host detail")
    error.__cause__ = socket.gaierror(-2, "private DNS detail")
    translated = gemini_service._translate_error(error)
    assert isinstance(translated, GeminiUnavailableError)
    assert translated.diagnostic_category == "DNS"
    assert "private" not in str(translated)


def test_model_argument_is_sdk_compatible_string() -> None:
    client = FakeGeminiClient([text_response("READY")])
    service = GeminiService(client=client, model="gemini-3.8-flash", max_retries=0)
    service.generate_content(contents="Reply READY only", system_instruction="test")
    assert client.calls[0]["model"] == "gemini-3.8-flash"
    assert isinstance(client.calls[0]["model"], str)


def test_rate_limits_retry_with_bounded_backoff() -> None:
    waits: list[float] = []
    client = FakeGeminiClient(
        [FakeAPIException(429), FakeAPIException(429), text_response("Ready.")]
    )
    service = GeminiService(
        client=client,
        max_retries=2,
        retry_backoff_seconds=0.1,
        sleep=waits.append,
    )
    response = service.generate_content(contents="hi", system_instruction="safe")
    assert response.candidates
    assert waits == [0.1, 0.2]
    assert len(client.calls) == 3


def test_authentication_failure_is_not_retried() -> None:
    client = FakeGeminiClient([FakeAPIException(401)])
    service = GeminiService(client=client, max_retries=3, sleep=lambda _: None)
    with pytest.raises(GeminiAuthenticationError):
        service.generate_content(contents="hi", system_instruction="safe")
    assert len(client.calls) == 1


def test_timeout_failure_is_retried() -> None:
    waits: list[float] = []
    client = FakeGeminiClient(
        [TimeoutError(), text_response("Ready.")]
    )
    service = GeminiService(
        client=client, max_retries=1, retry_backoff_seconds=0.01, sleep=waits.append
    )
    service.generate_content(contents="hi", system_instruction="safe")
    assert waits == [0.01]
    assert len(client.calls) == 2


def test_invalid_gemini_response_is_reported() -> None:
    assistant, _ = make_assistant([SimpleNamespace(candidates=[])])
    result = assistant.ask("Hello")
    assert result["status"] == "INVALID_MODEL_RESPONSE"


def test_tool_call_loop_is_bounded() -> None:
    responses = [
        function_response("get_simulation_status", {"scenario_id": "scenario-1"})
        for _ in range(4)
    ]
    assistant, client = make_assistant(responses, max_tool_rounds=2)
    result = assistant.ask("Status?")
    assert result["status"] == "TOOL_LIMIT_REACHED"
    assert len(client.calls) == 2


def test_unsupported_numeric_claim_is_not_returned_as_fact() -> None:
    assistant, _ = make_assistant(
        [
            function_response(
                "get_simulation_status", {"scenario_id": "scenario-1"}
            ),
            text_response("Exactly 42 victims were rescued."),
        ]
    )
    result = assistant.ask("How many victims have been rescued?")
    assert "42" not in result["reply"]
    assert result["tool_results"][0]["label"] == "MOCK"


def test_operational_answer_without_a_tool_result_is_not_asserted() -> None:
    assistant, _ = make_assistant([text_response("42 victims are safe.")])
    result = assistant.ask("How many victims have been rescued?")
    assert result["status"] == "NO_BACKEND_RESULT"
    assert "42" not in result["reply"]


def test_safe_logging_does_not_include_api_key_or_user_prompt(caplog: Any) -> None:
    secret = "AIza-test-secret-not-for-logs"
    client = FakeGeminiClient([FakeAPIException(401)])
    service = GeminiService(client=client, api_key=secret, max_retries=0)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(GeminiAuthenticationError):
            service.generate_content(
                contents="private user disaster details",
                system_instruction="safe",
            )
    assert secret not in caplog.text
    assert "private user disaster details" not in caplog.text


def test_missing_backend_does_not_return_fabricated_metrics() -> None:
    result = ToolRegistry(mode="real").execute(
        "get_simulation_status", {"scenario_id": "s-1"}
    )
    assert result["status"] == "SERVICE_UNAVAILABLE"
    assert result["data"] == {}
    assert "victims_rescued" not in result["data"]


def test_tool_registry_has_exact_allowlist() -> None:
    registry = ToolRegistry(mode="mock")
    assert {item["name"] for item in registry.function_declarations} == set(TOOL_NAMES)
