"""Run a credential-free mock conversation through the assistant tool loop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ai_agent import AIAssistant
from ai_tools import ToolRegistry
from gemini_service import GeminiService


def _function_call(name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name=name, args=arguments, id="offline-call"
                            )
                        )
                    ]
                )
            )
        ]
    )


def _text(text: str) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))
        ]
    )


class OfflineGeminiClient:
    """A tiny deterministic client fake; it performs no network requests."""

    def __init__(self) -> None:
        self.models = self
        self._responses = [
            _function_call(
                "get_flood_risk", {"hours": 10, "scenario_id": "demo-scenario"}
            ),
            _text(
                "The MOCK fixture has no flood assessment or forecast for "
                "demo-scenario at the requested horizon."
            ),
            _function_call(
                "get_flood_risk", {"hours": 5, "scenario_id": "demo-scenario"}
            ),
            _text("A follow-up check is available as a MOCK fixture only."),
        ]

    def generate_content(self, **_: Any) -> Any:
        if not self._responses:
            return _text("No additional offline demo response is configured.")
        return self._responses.pop(0)


def main() -> None:
    assistant = AIAssistant(
        gemini_service=GeminiService(client=OfflineGeminiClient(), max_retries=0),
        tool_registry=ToolRegistry(
            mode="mock", default_scenario_id="demo-scenario"
        ),
    )
    response = assistant.ask("Show flood risk after 10 hours.")
    print(f"Assistant: {response['reply']}")
    print(f"Tool execution: {response['tools_executed']}")
    print(f"Backend mode: {response.get('backend_mode', 'not reported')}")
    followup = assistant.ask(
        "What about after 5 hours?", session_id=response["session_id"]
    )
    print(f"Follow-up: {followup['reply']}")
    print(f"Follow-up tool execution: {followup['tools_executed']}")


if __name__ == "__main__":
    main()
