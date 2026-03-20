from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5-mini"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
DEFAULT_PROMPT_NAME = "system_default.md"
LOCAL_CONFIG_NAME = "llm_config.local.json"
DEFAULT_DEBUG_LOG_NAME = "llm_chat_debug.jsonl"
_DEBUG_LOG_LOCK = threading.Lock()

_MISSING = object()

# LLM setup state (``setup_llm`` / ``POST /api/llm/config``). Highest priority in ``from_env``.
_LLM_SETUP_LOCK = threading.Lock()
_llm_setup_config_path: str | None = None
_llm_setup_patch: dict[str, Any] = {}

# Keys shared with ``llm_config.local.json`` / ``llm_config.example.json`` / env vars.
LLM_SETUP_KEYS = frozenset(
    {
        "base_url",
        "api_key",
        "model",
        "timeout_sec",
        "max_tokens",
        "extra_headers",
        "debug_log_enabled",
        "debug_log_path",
    }
)


def clear_llm_setup() -> None:
    """Clear setup file path and in-memory field patches (fall back to env + local JSON only)."""
    global _llm_setup_config_path, _llm_setup_patch
    with _LLM_SETUP_LOCK:
        _llm_setup_config_path = None
        _llm_setup_patch = {}


def setup_llm(*, config_path: Any = _MISSING, clear: bool = False, **kwargs: Any) -> None:
    """
    Configure the visualizer LLM client (in-process). Call **before** ``launch()``, or use
    ``POST /api/llm/config`` for the same fields (HTTP API does **not** accept ``config_path``
    for security).

    :param config_path: Optional JSON file (same shape as ``llm_config.example.json``). Merged
        after ``llm_config.local.json`` and before environment variables.
    :param clear: If True, ignore other arguments and reset setup state entirely.
    :param kwargs: Any keys in ``LLM_SETUP_KEYS``. Pass ``None`` or ``\"\"`` for string fields
        to remove that key from the patch so lower layers apply.

    Final resolution order (each step overwrites defined, non-blank values from the previous):

    1. Built-in defaults
    2. ``llm_config.local.json`` (optional)
    3. JSON file from ``config_path`` (optional)
    4. Environment variables (``TRITON_VIZ_LLM_*``, ``OPENAI_API_KEY``)
    5. Keyword arguments passed here (or JSON body for ``POST /api/llm/config``) — highest
    """
    global _llm_setup_config_path, _llm_setup_patch
    with _LLM_SETUP_LOCK:
        if clear:
            _llm_setup_config_path = None
            _llm_setup_patch = {}
            return

        if config_path is not _MISSING:
            if config_path is None or (
                isinstance(config_path, str) and not str(config_path).strip()
            ):
                _llm_setup_config_path = None
            else:
                _llm_setup_config_path = os.path.abspath(
                    os.path.expanduser(str(config_path).strip())
                )

        bad = set(kwargs) - LLM_SETUP_KEYS
        if bad:
            raise TypeError(
                f"setup_llm: unknown keyword argument(s): {sorted(bad)}. "
                f"Allowed: {sorted(LLM_SETUP_KEYS)}"
            )

        for k, v in kwargs.items():
            if k == "extra_headers":
                if v is None:
                    _llm_setup_patch.pop(k, None)
                elif isinstance(v, dict):
                    _llm_setup_patch[k] = {
                        str(x): str(y)
                        for x, y in v.items()
                        if x is not None and y is not None
                    }
                else:
                    raise TypeError("extra_headers must be a dict or None")
                continue
            if v is None or (isinstance(v, str) and not str(v).strip()):
                _llm_setup_patch.pop(k, None)
            elif k in ("debug_log_enabled",):
                _llm_setup_patch[k] = v
            elif k == "max_tokens":
                try:
                    _llm_setup_patch[k] = int(v)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"setup_llm: max_tokens must be int-compatible, got {v!r}"
                    ) from exc
            elif k == "timeout_sec":
                try:
                    _llm_setup_patch[k] = float(v)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"setup_llm: timeout_sec must be float-compatible, got {v!r}"
                    ) from exc
            elif isinstance(v, str):
                _llm_setup_patch[k] = str(v).strip()
            else:
                _llm_setup_patch[k] = v


def setup_llm_from_file(path: str) -> None:
    """Convenience: ``setup_llm(config_path=path)``."""
    setup_llm(config_path=path)


def _get_llm_setup_snapshot() -> tuple[str | None, dict[str, Any]]:
    with _LLM_SETUP_LOCK:
        return _llm_setup_config_path, dict(_llm_setup_patch)


def _load_llm_config_at_path(path: str | None) -> dict[str, Any]:
    """Load a JSON LLM config file; never raises; returns {} if missing or invalid."""
    if not path:
        return {}
    try:
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _llm_env_layer() -> dict[str, Any]:
    """Environment variables as a config layer (between setup file and setup patch)."""
    layer: dict[str, Any] = {}
    if v := _str_or_none(os.getenv("TRITON_VIZ_LLM_BASE_URL")):
        layer["base_url"] = v
    if v := _str_or_none(os.getenv("TRITON_VIZ_LLM_API_KEY")):
        layer["api_key"] = v
    elif v := _str_or_none(os.getenv("OPENAI_API_KEY")):
        layer["api_key"] = v
    if v := _str_or_none(os.getenv("TRITON_VIZ_LLM_MODEL")):
        layer["model"] = v
    if os.getenv("TRITON_VIZ_LLM_TIMEOUT") is not None:
        layer["timeout_sec"] = os.getenv("TRITON_VIZ_LLM_TIMEOUT")
    if os.getenv("TRITON_VIZ_LLM_MAX_TOKENS") is not None:
        layer["max_tokens"] = os.getenv("TRITON_VIZ_LLM_MAX_TOKENS")
    if os.getenv("TRITON_VIZ_LLM_DEBUG_LOG") is not None:
        layer["debug_log_enabled"] = os.getenv("TRITON_VIZ_LLM_DEBUG_LOG")
    if os.getenv("TRITON_VIZ_LLM_DEBUG_LOG_PATH") is not None:
        layer["debug_log_path"] = os.getenv("TRITON_VIZ_LLM_DEBUG_LOG_PATH")
    return layer


def _apply_llm_config_layer(target: dict[str, Any], layer: dict[str, Any]) -> None:
    """Merge one JSON-style layer into *target* (only ``LLM_SETUP_KEYS``)."""
    for k, v in layer.items():
        if k not in LLM_SETUP_KEYS:
            continue
        if k == "extra_headers":
            if isinstance(v, dict) and v:
                target["extra_headers"] = {
                    str(x): str(y)
                    for x, y in v.items()
                    if x is not None and y is not None
                }
            continue
        if v is None:
            continue
        if isinstance(v, str) and not v.strip() and k != "debug_log_path":
            continue
        target[k] = v


def _build_llm_merged_dict() -> dict[str, Any]:
    """Merge defaults, local JSON, setup file, env, and setup patch into one dict."""
    setup_path, patch = _get_llm_setup_snapshot()
    file_cfg = _load_llm_config_at_path(setup_path)
    local_cfg = _load_local_llm_config()
    env_layer = _llm_env_layer()

    merged: dict[str, Any] = {
        "base_url": DEFAULT_BASE_URL,
        "api_key": None,
        "model": DEFAULT_MODEL,
        "timeout_sec": 60.0,
        "max_tokens": 2048,
        "extra_headers": {},
        "debug_log_enabled": False,
        "debug_log_path": os.path.join(os.path.dirname(__file__), DEFAULT_DEBUG_LOG_NAME),
    }
    for layer in (local_cfg, file_cfg, env_layer, patch):
        _apply_llm_config_layer(merged, layer)
    return merged


class LLMAPIError(RuntimeError):
    """Raised when an LLM API call fails."""


def load_prompt_template(name: str = DEFAULT_PROMPT_NAME) -> str:
    """Load a prompt template from visualizer/prompts."""
    safe_name = os.path.basename(name or DEFAULT_PROMPT_NAME)
    path = os.path.join(PROMPTS_DIR, safe_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {safe_name}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _str_or_none(value: Any) -> str | None:
    """Return stripped string or None if missing / blank (treats placeholders as empty)."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _load_local_llm_config() -> dict[str, Any]:
    """
    Load optional local config from visualizer/llm_config.local.json.
    Returns empty dict when file is missing or invalid (never raises).
    """
    config_path = os.path.join(os.path.dirname(__file__), LOCAL_CONFIG_NAME)
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _resolve_debug_log_path(path_value: Any) -> str:
    text = str(path_value or "").strip()
    if not text:
        return os.path.join(os.path.dirname(__file__), DEFAULT_DEBUG_LOG_NAME)
    if os.path.isabs(text):
        return text
    return os.path.join(os.path.dirname(__file__), text)


@dataclass
class OpenAICompatibleConfig:
    """
    Configuration for OpenAI-compatible Chat Completions API.

    Built by ``from_env()`` using (lowest → highest priority):

    1. Built-in defaults
    2. ``llm_config.local.json`` next to this module (optional; missing file is OK)
    3. JSON file path from ``setup_llm(config_path=...)`` (optional)
    4. Environment variables — ``TRITON_VIZ_LLM_*``, ``OPENAI_API_KEY``
    5. ``setup_llm(**kwargs)`` or ``POST /api/llm/config`` (same keys as the JSON file)

    Missing files or invalid JSON never raise; blank values in lower layers are skipped.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    timeout_sec: float = 60.0
    max_tokens: int = 2048
    extra_headers: dict[str, str] = field(default_factory=dict)
    debug_log_enabled: bool = False
    debug_log_path: str = os.path.join(os.path.dirname(__file__), DEFAULT_DEBUG_LOG_NAME)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        merged = _build_llm_merged_dict()

        base_url_raw = _str_or_none(merged.get("base_url")) or DEFAULT_BASE_URL
        base_url = _normalize_base_url(base_url_raw)
        api_key = _str_or_none(merged.get("api_key"))
        model = _str_or_none(merged.get("model")) or DEFAULT_MODEL

        try:
            timeout_sec = float(merged.get("timeout_sec", 60.0))
        except (TypeError, ValueError):
            timeout_sec = 60.0

        try:
            max_tokens = max(1, int(merged.get("max_tokens", 2048)))
        except (TypeError, ValueError):
            max_tokens = 2048

        extra = merged.get("extra_headers") or {}
        extra_headers: dict[str, str] = {}
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key is None or value is None:
                    continue
                extra_headers[str(key)] = str(value)

        debug_log_enabled = _to_bool(merged.get("debug_log_enabled", False))
        debug_log_path = _resolve_debug_log_path(merged.get("debug_log_path"))

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            debug_log_enabled=debug_log_enabled,
            debug_log_path=debug_log_path,
        )


class OpenAICompatibleClient:
    """
    Lightweight wrapper for OpenAI-compatible Chat Completions APIs.
    """

    def __init__(self, config: OpenAICompatibleConfig | None = None):
        self.config = config or OpenAICompatibleConfig.from_env()
        self.base_url = _normalize_base_url(self.config.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _post_chat_completions(self, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        try:
            return requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.config.timeout_sec,
            )
        except requests.RequestException as exc:
            raise LLMAPIError(f"LLM request failed: {exc}") from exc

    def append_debug_log(self, event: dict[str, Any]) -> None:
        """
        Append one JSONL debug event when debug logging is enabled.
        Never raises to avoid affecting user requests.
        """
        if not self.config.debug_log_enabled:
            return
        try:
            line = dict(event)
            line["ts"] = datetime.now(timezone.utc).isoformat()
            path = self.config.debug_log_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _DEBUG_LOG_LOCK:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            return

    @staticmethod
    def _truncate_text(value: Any, limit: int = 2000) -> str:
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    @staticmethod
    def extract_answer_and_debug(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Extract answer text from OpenAI-compatible response with diagnostics.
        Supports both 'text' and 'output_text' content blocks.
        """
        choices = data.get("choices") or []
        usage = data.get("usage")
        debug: dict[str, Any] = {
            "choices_count": len(choices),
            "usage": usage,
            "finish_reason": None,
            "content_kind": None,
            "content_block_types": [],
            "refusal": None,
            "message_preview": None,
        }
        if not choices:
            return "", debug

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        debug["finish_reason"] = first_choice.get("finish_reason")
        message = first_choice.get("message") or {}
        content = message.get("content")

        # String content
        if isinstance(content, str):
            debug["content_kind"] = "string"
            debug["message_preview"] = OpenAICompatibleClient._truncate_text(content)
            return content, debug

        # Block content
        if isinstance(content, list):
            debug["content_kind"] = "list"
            text_parts: list[str] = []
            block_types: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "")
                if block_type:
                    block_types.append(block_type)
                if block_type in {"text", "output_text"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif block_type == "refusal":
                    refusal_text = block.get("refusal")
                    if isinstance(refusal_text, str):
                        debug["refusal"] = refusal_text
            debug["content_block_types"] = block_types
            answer = "".join(text_parts)
            debug["message_preview"] = OpenAICompatibleClient._truncate_text(
                json.dumps(message, ensure_ascii=False)
            )
            if answer:
                return answer, debug

        # Fallback: explicit refusal field or empty
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            debug["refusal"] = refusal
        debug["message_preview"] = OpenAICompatibleClient._truncate_text(
            json.dumps(message, ensure_ascii=False)
        )
        return "", debug

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call POST /chat/completions and return parsed JSON.
        """
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        # gpt-5-mini only supports default temperature behavior.
        # Only pass temperature when explicitly set to 1.
        if temperature is not None and float(temperature) == 1.0:
            payload["temperature"] = 1
        # For gpt-5-mini on Chat Completions, use max_completion_tokens.
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        response = self._post_chat_completions(payload)

        if not response.ok:
            message = response.text
            raise LLMAPIError(
                f"LLM request returned {response.status_code}: {message}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise LLMAPIError("LLM response is not valid JSON") from exc

    def chat_completions_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """
        Convenience helper to return first text content from choices.
        """
        data = self.chat_completions(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        answer, _debug = self.extract_answer_and_debug(data)
        return answer
