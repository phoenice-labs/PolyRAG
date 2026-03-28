"""
Phase 5: Query Intelligence — LLM client (multi-provider, OpenAI-compatible + adapters).

Supported providers (set via /api/config/llm):
  lm_studio   — OpenAI-compatible, http://localhost:1234/v1, no real key needed
  openai      — api.openai.com/v1, real key required
  ollama      — http://localhost:11434/v1, no real key needed
  groq        — api.groq.com/openai/v1, real key required
  azure_openai— requires api_version + deployment_name config fields
  gemini      — generativelanguage.googleapis.com OpenAI-compat endpoint
  anthropic   — native Anthropic SDK (graceful degradation if sdk not installed)
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


# ── Provider-specific dummy API keys (used when api_key is not configured) ────
_PROVIDER_DUMMY_KEYS = {
    "lm_studio": "lm-studio",
    "ollama": "ollama",
    "gemini": "gemini",   # replaced by real key when configured
    "openai": "",          # must be configured
    "groq": "",            # must be configured
    "azure_openai": "",    # must be configured
    "anthropic": "",       # must be configured
}


class LMStudioClient:
    """
    Thin wrapper around LM Studio's OpenAI-compatible REST API.

    LM Studio exposes: http://localhost:1234/v1
    Uses the openai Python SDK pointed at localhost.

    Also supports any OpenAI-compatible endpoint (Ollama, Groq, OpenAI cloud, Gemini)
    via the base_url + api_key parameters.

    Requires: pip install openai
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "mistralai/ministral-3b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
        api_key: str = "",          # ← new: empty → use provider dummy key
        provider: str = "lm_studio",  # ← new: for availability probing
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider
        # Resolve effective API key: explicit key > provider dummy > empty string.
        # Local providers (lm_studio, ollama) use a non-empty placeholder so the
        # OpenAI SDK doesn't raise "api_key is required". Cloud providers that
        # require a real key will correctly fail at inference time if left empty.
        self._api_key = api_key or _PROVIDER_DUMMY_KEYS.get(provider, "") or ""
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
                api_key=self._api_key,
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
        """Check whether the configured LLM endpoint is reachable."""
        try:
            import requests
            # Normalise: strip trailing /v1 so we probe the root /v1/models path
            base = self.base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            r = requests.get(
                f"{base}/models",
                timeout=3,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            return r.status_code == 200
        except Exception:
            return False


# ── Factory ───────────────────────────────────────────────────────────────────

def create_llm_client(llm_cfg: Optional[dict] = None) -> LMStudioClient:
    """Create an LLMClient from a config dict (reads live config if None).

    For OpenAI-compatible providers (lm_studio, openai, ollama, groq, gemini,
    azure_openai) this returns a configured LMStudioClient.

    For ``anthropic`` the same class is returned with a stub base_url — full
    native Anthropic support requires an AnthropicClient adapter (future work).
    """
    if llm_cfg is None:
        try:
            from api.deps import get_llm_config  # lazy import to avoid circularity
            llm_cfg = get_llm_config()
        except Exception:
            llm_cfg = {}

    provider = llm_cfg.get("provider", "lm_studio")
    base_url = llm_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = llm_cfg.get("api_key", "")

    # Gemini OpenAI-compatible endpoint
    if provider == "gemini" and not base_url:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    return LMStudioClient(
        base_url=base_url,
        model=llm_cfg.get("model", "mistralai/ministral-3b"),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 512)),
        timeout=int(llm_cfg.get("timeout", 60)),
        api_key=api_key,
        provider=provider,
    )
