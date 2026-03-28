"""
/api/config — LLM provider configuration endpoints.

GET  /api/config/llm         → return current LLM config (api_key masked)
PUT  /api/config/llm         → update + persist LLM config
GET  /api/config/llm/test    → probe the configured LLM endpoint
GET  /api/config/llm/providers → list supported providers with metadata
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_llm_config, save_llm_config

router = APIRouter(tags=["config"])

# ── Supported providers ───────────────────────────────────────────────────────

_PROVIDERS: List[Dict[str, Any]] = [
    {
        "id": "lm_studio",
        "label": "LM Studio",
        "protocol": "openai_compatible",
        "default_base_url": "http://localhost:1234/v1",
        "requires_api_key": False,
        "notes": "Run models locally via LM Studio desktop app.",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "protocol": "openai_native",
        "default_base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
        "notes": "GPT-4o, GPT-4-turbo, etc. Real API key required.",
    },
    {
        "id": "ollama",
        "label": "Ollama",
        "protocol": "openai_compatible",
        "default_base_url": "http://localhost:11434/v1",
        "requires_api_key": False,
        "notes": "Run open-source models locally via Ollama.",
    },
    {
        "id": "groq",
        "label": "Groq",
        "protocol": "openai_compatible",
        "default_base_url": "https://api.groq.com/openai/v1",
        "requires_api_key": True,
        "notes": "Fast inference for open-source models. Real API key required.",
    },
    {
        "id": "azure_openai",
        "label": "Azure OpenAI",
        "protocol": "openai_compatible",
        "default_base_url": "",
        "requires_api_key": True,
        "notes": "Azure-hosted OpenAI models. Set base_url to your deployment endpoint.",
    },
    {
        "id": "gemini",
        "label": "Google Gemini",
        "protocol": "openai_compatible",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "requires_api_key": True,
        "notes": "Gemini via OpenAI-compatible endpoint. Real API key required.",
    },
    {
        "id": "anthropic",
        "label": "Anthropic Claude",
        "protocol": "anthropic_native",
        "default_base_url": "https://api.anthropic.com",
        "requires_api_key": True,
        "notes": "Claude models. Real API key required. Native SDK support in roadmap.",
    },
]

_PROVIDER_IDS = {p["id"] for p in _PROVIDERS}

# Dummy auth tokens for local providers that don't require real API keys.
# Cloud providers get an empty string — they will correctly return 401 if no real key is set.
_PROVIDER_DUMMY_KEYS = {p["id"]: p["id"].replace("_", "-") for p in _PROVIDERS if not p["requires_api_key"]}


# ── Schemas ───────────────────────────────────────────────────────────────────

class LLMConfigRequest(BaseModel):
    provider: str = Field(default="lm_studio", description="LLM provider ID")
    base_url: str = Field(default="http://localhost:1234/v1", description="API base URL")
    api_key: str = Field(default="", description="API key (stored locally, never committed)")
    model: str = Field(default="mistralai/ministral-3b", description="Model name/ID")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=64, le=8192)
    timeout: int = Field(default=60, ge=5, le=600)

    model_config = {"extra": "ignore"}


class LLMConfigResponse(BaseModel):
    provider: str
    base_url: str
    api_key_set: bool       # True if a non-empty key is stored (key value is never returned)
    model: str
    temperature: float
    max_tokens: int
    timeout: int


class LLMTestResponse(BaseModel):
    reachable: bool
    provider: str
    base_url: str
    model: str
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask(cfg: Dict[str, Any]) -> LLMConfigResponse:
    return LLMConfigResponse(
        provider=cfg.get("provider", "lm_studio"),
        base_url=cfg.get("base_url", ""),
        api_key_set=bool(cfg.get("api_key", "")),
        model=cfg.get("model", ""),
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 512),
        timeout=cfg.get("timeout", 60),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/config/llm/providers")
def list_providers() -> List[Dict[str, Any]]:
    """List all supported LLM providers with metadata."""
    return _PROVIDERS


@router.get("/config/llm", response_model=LLMConfigResponse)
def get_llm_config_endpoint() -> LLMConfigResponse:
    """Return the current LLM configuration. API key value is never exposed."""
    return _mask(get_llm_config())


@router.put("/config/llm", response_model=LLMConfigResponse)
def update_llm_config(body: LLMConfigRequest) -> LLMConfigResponse:
    """Update the LLM configuration and flush the pipeline cache.

    The new config is persisted to data/llm_config.json (not committed to git).
    All subsequent requests will use the new provider/model immediately.

    Security note: api_key is stored only in the local data directory, never
    in source control. Use environment variables for production deployments.
    """
    if body.provider not in _PROVIDER_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider '{body.provider}'. Valid: {sorted(_PROVIDER_IDS)}",
        )
    new_cfg = body.model_dump()
    save_llm_config(new_cfg)
    return _mask(new_cfg)


@router.get("/config/llm/test", response_model=LLMTestResponse)
async def test_llm_connection() -> LLMTestResponse:
    """Probe the currently configured LLM endpoint and return reachability status."""
    cfg = get_llm_config()
    provider = cfg.get("provider", "lm_studio")
    base_url = cfg.get("base_url", "")
    model = cfg.get("model", "")
    api_key = cfg.get("api_key", "") or _PROVIDER_DUMMY_KEYS.get(provider, "")

    try:
        import requests as _req
        # Normalise: ensure we probe /v1/models
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        resp = _req.get(
            f"{url}/models",
            timeout=5,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        reachable = resp.status_code == 200
        error = None if reachable else f"HTTP {resp.status_code}"
    except Exception as exc:
        reachable = False
        error = str(exc)

    return LLMTestResponse(
        reachable=reachable,
        provider=provider,
        base_url=base_url,
        model=model,
        error=error,
    )
