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


def _load_local_llm_config() -> dict[str, Any]:
    """
    Load optional local config from visualizer/llm_config.local.json.
    Returns empty dict when file is missing or invalid.
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

    Environment variable fallbacks:
    - TRITON_VIZ_LLM_BASE_URL (default: https://api.openai.com/v1)
    - TRITON_VIZ_LLM_API_KEY (fallback: OPENAI_API_KEY)
    - TRITON_VIZ_LLM_MODEL (default: gpt-5-mini)
    - TRITON_VIZ_LLM_TIMEOUT (seconds, default: 60)
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
        local_cfg = _load_local_llm_config()

        base_url = (
            os.getenv("TRITON_VIZ_LLM_BASE_URL")
            or local_cfg.get("base_url")
            or DEFAULT_BASE_URL
        )
        api_key = (
            os.getenv("TRITON_VIZ_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or local_cfg.get("api_key")
        )
        model = os.getenv("TRITON_VIZ_LLM_MODEL") or local_cfg.get("model") or DEFAULT_MODEL
        timeout_raw = (
            os.getenv("TRITON_VIZ_LLM_TIMEOUT")
            or str(local_cfg.get("timeout_sec", "60"))
        )
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 60.0
        max_tokens_raw = os.getenv(
            "TRITON_VIZ_LLM_MAX_TOKENS", str(local_cfg.get("max_tokens", 2048))
        )
        try:
            max_tokens = max(1, int(max_tokens_raw))
        except (TypeError, ValueError):
            max_tokens = 2048

        extra_headers_raw = local_cfg.get("extra_headers")
        extra_headers: dict[str, str] = {}
        if isinstance(extra_headers_raw, dict):
            for key, value in extra_headers_raw.items():
                if key is None or value is None:
                    continue
                extra_headers[str(key)] = str(value)

        debug_log_enabled = _to_bool(
            os.getenv("TRITON_VIZ_LLM_DEBUG_LOG", local_cfg.get("debug_log_enabled", False))
        )
        debug_log_path = _resolve_debug_log_path(
            os.getenv("TRITON_VIZ_LLM_DEBUG_LOG_PATH", local_cfg.get("debug_log_path"))
        )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_sec=timeout,
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
