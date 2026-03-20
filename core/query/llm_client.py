"""
Phase 5: Query Intelligence — LM Studio client (OpenAI-compatible API).
Model: mistralai/ministral-3b running at localhost:1234 via LM Studio.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Persistent LLM trace log — survives process restarts.
_TRACE_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "llm_traces.jsonl"
_trace_log_lock = threading.Lock()


def _append_trace_to_log(entry: "LLMTraceEntry", timestamp: str) -> None:
    """Append one LLM trace as a JSONL line to the persistent log file."""
    try:
        _TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": timestamp,
            "method": entry.method,
            "system_prompt": entry.system_prompt,
            "user_message": entry.user_message,
            "response": entry.response,
            "latency_ms": entry.latency_ms,
        }
        with _trace_log_lock:
            with _TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # log failures must never crash the search path


@dataclass
class LLMTraceEntry:
    """One LLM call captured during a search request."""
    method: str          # human-readable retrieval method name
    system_prompt: str   # system prompt sent to LLM
    user_message: str    # user/context message sent to LLM
    response: str        # raw LLM response text
    latency_ms: float    # round-trip latency in milliseconds


class LMStudioClient:
    """
    Thin wrapper around LM Studio's OpenAI-compatible REST API.

    LM Studio exposes: http://localhost:1234/v1
    Uses the openai Python SDK pointed at localhost.

    Requires: pip install openai
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "mistralai/ministral-3b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None
        self._traces: List[LLMTraceEntry] = []

    def begin_request(self) -> None:
        """Clear accumulated traces — call at the start of each search request."""
        self._traces = []

    def get_traces(self) -> List[LLMTraceEntry]:
        """Return all LLM traces captured since the last begin_request() call."""
        return list(self._traces)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("Install openai: pip install openai") from e
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="lm-studio",   # LM Studio does not require a real key
                timeout=self.timeout,
            )
        return self._client

    def complete(self, prompt: str, system: str = "",
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 trace_method: str = "") -> str:
        """Send a completion request and return the response text.

        Parameters
        ----------
        trace_method : human-readable name of the retrieval method making this call.
                       When non-empty the call is captured in self._traces.
        """
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result = response.choices[0].message.content.strip()

        if trace_method:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            entry = LLMTraceEntry(
                method=trace_method,
                system_prompt=system,
                user_message=prompt,
                response=result,
                latency_ms=latency_ms,
            )
            self._traces.append(entry)
            _append_trace_to_log(entry, ts)   # ← persist to disk immediately

        return result

    def is_available(self) -> bool:
        """Check whether LM Studio is reachable (used to skip tests gracefully)."""
        try:
            import requests
            r = requests.get(
                f"{self.base_url.rstrip('/v1')}/v1/models",
                timeout=3,
                headers={"Authorization": "Bearer lm-studio"},
            )
            return r.status_code == 200
        except Exception:
            return False
