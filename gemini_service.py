"""Gemini API boundary for the RescueBot assistant.

The service does not make a request until ``generate_content`` is called.
Clients can be injected to keep tests and offline examples credential-free.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import httpx

logger = logging.getLogger(__name__)


class GeminiServiceError(RuntimeError):
    """Base class for sanitized Gemini service errors."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_category: str = "UNKNOWN",
        http_status: int | None = None,
        diagnostic_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_category = diagnostic_category
        self.http_status = http_status
        self.diagnostic_reason = diagnostic_reason


class GeminiConfigurationError(GeminiServiceError):
    """Raised when required Gemini configuration is missing or invalid."""


class GeminiAuthenticationError(GeminiServiceError):
    """Raised when Gemini rejects the configured credentials."""


class GeminiRateLimitError(GeminiServiceError):
    """Raised when the account's Gemini quota or rate limit is exhausted."""


class GeminiTimeoutError(GeminiServiceError):
    """Raised when Gemini does not respond before the configured timeout."""


class GeminiInvalidResponseError(GeminiServiceError):
    """Raised when Gemini returns no usable response candidates."""


class GeminiUnavailableError(GeminiServiceError):
    """Raised when Gemini is temporarily unavailable or fails a request."""


class GeminiRequestError(GeminiServiceError):
    """Raised for non-retryable Gemini request failures."""


def _sanitized_error(
    error_type: type[GeminiServiceError],
    message: str,
    category: str,
    http_status: int | None = None,
    diagnostic_reason: str | None = None,
) -> GeminiServiceError:
    return error_type(
        message,
        diagnostic_category=category,
        http_status=http_status,
        diagnostic_reason=diagnostic_reason,
    )


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _safe_google_reason(error: Exception) -> str | None:
    details = getattr(error, "details", None)
    if not isinstance(details, Mapping):
        return None

    error_details = details.get("error", details)
    if not isinstance(error_details, Mapping):
        return None

    structured_details = error_details.get("details", [])
    if isinstance(structured_details, list):
        for detail in structured_details:
            if not isinstance(detail, Mapping):
                continue
            reason = detail.get("reason")
            if isinstance(reason, str) and re.fullmatch(
                r"[A-Z][A-Z0-9_]{0,63}", reason
            ):
                return reason

    status = error_details.get("status")
    if isinstance(status, str) and re.fullmatch(
        r"[A-Z][A-Z0-9_]{0,63}", status
    ):
        return status
    return None


def _exception_chain(error: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _network_category(error: Exception) -> str | None:
    chain = _exception_chain(error)
    if any(isinstance(item, httpx.TimeoutException) for item in chain):
        return "TIMEOUT"
    if any(isinstance(item, httpx.ProxyError) for item in chain):
        return "PROXY"
    if any(isinstance(item, socket.gaierror) for item in chain):
        return "DNS"
    if any(isinstance(item, (httpx.ConnectError, httpx.NetworkError, OSError)) for item in chain):
        return "NETWORK"
    if "timeout" in type(error).__name__.lower():
        return "TIMEOUT"
    return None


def _translate_error(error: Exception) -> GeminiServiceError:
    if isinstance(error, GeminiServiceError):
        return error
    status = _status_code(error)
    diagnostic_reason = _safe_google_reason(error)

    network_category = _network_category(error)
    if network_category == "TIMEOUT":
        return _sanitized_error(
            GeminiTimeoutError,
            "Gemini request timed out.",
            network_category,
            status,
            diagnostic_reason,
        )
    if network_category in {"DNS", "PROXY", "NETWORK"}:
        return _sanitized_error(
            GeminiUnavailableError,
            "Gemini could not be reached.",
            network_category,
            status,
            diagnostic_reason,
        )

    reason_categories = {
        "API_KEY_INVALID": "AUTHENTICATION",
        "API_KEY_SERVICE_BLOCKED": "API_KEY_RESTRICTED",
        "MODEL_NOT_FOUND": "MODEL_NOT_FOUND",
        "PERMISSION_DENIED": "ACCESS_DENIED",
        "RESOURCE_EXHAUSTED": "QUOTA_OR_RATE_LIMIT",
        "SERVICE_DISABLED": "API_NOT_ENABLED",
    }
    reason_category = reason_categories.get(diagnostic_reason or "")

    if status == 401:
        return _sanitized_error(
            GeminiAuthenticationError,
            "Gemini authentication failed.",
            reason_category or "AUTHENTICATION",
            status,
            diagnostic_reason,
        )
    if status == 403:
        return _sanitized_error(
            GeminiAuthenticationError,
            "Gemini access was denied.",
            reason_category or "ACCESS_DENIED",
            status,
            diagnostic_reason,
        )
    if status == 429:
        return _sanitized_error(
            GeminiRateLimitError,
            "Gemini quota or rate limit was reached.",
            reason_category or "QUOTA_OR_RATE_LIMIT",
            status,
            diagnostic_reason,
        )
    if status in (408, 504):
        return _sanitized_error(
            GeminiTimeoutError,
            "Gemini request timed out.",
            "TIMEOUT",
            status,
            diagnostic_reason,
        )
    if status is not None and status >= 500:
        return _sanitized_error(
            GeminiUnavailableError,
            "Gemini service is temporarily unavailable.",
            "SERVICE_UNAVAILABLE",
            status,
            diagnostic_reason,
        )
    if isinstance(error, errors.APIError):
        category = reason_category or (
            "MODEL_NOT_FOUND" if status == 404 else "API_REQUEST"
        )
        return _sanitized_error(
            GeminiRequestError,
            "Gemini rejected the request.",
            category,
            status,
            diagnostic_reason,
        )
    if isinstance(error, errors.UnknownApiResponseError):
        return _sanitized_error(
            GeminiRequestError,
            "Gemini returned an unrecognized API response.",
            "API_RESPONSE",
            status,
            diagnostic_reason,
        )
    if isinstance(error, (httpx.InvalidURL, ValueError)):
        return _sanitized_error(
            GeminiConfigurationError,
            "Gemini request configuration is invalid.",
            "CLIENT_CONFIGURATION",
            status,
            diagnostic_reason,
        )
    return _sanitized_error(
        GeminiRequestError,
        "Gemini request failed.",
        "SDK_ERROR",
        status,
        diagnostic_reason,
    )


class GeminiService:
    """Reusable, bounded-retry wrapper around the official Google GenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        timeout_seconds: float = 20.0,
        sleep: Any = time.sleep,
    ) -> None:
        load_dotenv(override=False)
        self._api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
        self._client = client
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep

        if not self.model:
            raise GeminiConfigurationError(
                "GEMINI_MODEL must not be empty.",
                diagnostic_category="CLIENT_CONFIGURATION",
            )
        if max_retries < 0 or retry_backoff_seconds < 0 or timeout_seconds <= 0:
            raise GeminiConfigurationError(
                "Gemini retry and timeout settings are invalid.",
                diagnostic_category="CLIENT_CONFIGURATION",
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is required before using Gemini.",
                diagnostic_category="MISSING_CREDENTIALS",
            )
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )
        return self._client

    def generate_content(
        self,
        *,
        contents: Any,
        function_declarations: Sequence[dict[str, Any]] = (),
        system_instruction: str,
    ) -> Any:
        """Request one model turn, leaving function execution to the caller."""
        declarations = [
            types.FunctionDeclaration(
                name=declaration["name"],
                description=declaration["description"],
                parameters_json_schema=declaration["parameters"],
            )
            for declaration in function_declarations
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        client = self._get_client()

        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as error:
                translated = _translate_error(error)
                retryable = isinstance(
                    translated,
                    (GeminiRateLimitError, GeminiTimeoutError, GeminiUnavailableError),
                )
                if retryable and attempt < self.max_retries:
                    logger.warning(
                        "Gemini request failed: category=%s http_status=%s "
                        "reason=%s; retry %d of %d.",
                        translated.diagnostic_category,
                        translated.http_status,
                        translated.diagnostic_reason,
                        attempt + 1,
                        self.max_retries,
                    )
                    self._sleep(
                        min(self.retry_backoff_seconds * (2**attempt), 4.0)
                    )
                    continue
                logger.warning(
                    "Gemini request failed: category=%s http_status=%s reason=%s.",
                    translated.diagnostic_category,
                    translated.http_status,
                    translated.diagnostic_reason,
                )
                raise translated from error

            candidates = getattr(response, "candidates", None)
            if not candidates:
                logger.warning("Gemini returned no response candidates.")
                raise GeminiInvalidResponseError(
                    "Gemini returned no usable response candidates.",
                    diagnostic_category="INVALID_RESPONSE",
                )
            return response

        raise GeminiUnavailableError(
            "Gemini request failed after bounded retries.",
            diagnostic_category="RETRY_LIMIT",
        )
