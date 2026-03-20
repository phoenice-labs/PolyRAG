"""
Prompt management router: GET/PUT /api/prompts, POST /api/prompts/{key}/reset

Allows the PromptEditor UI to read and update LLM prompts stored in
config/prompts.yaml without any code changes or server restarts.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.prompt_registry import PromptRegistry

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptUpdateRequest(BaseModel):
    template: str


class PromptEntry(BaseModel):
    key: str
    method_name: str
    method_id: int
    pipeline_stage: str
    description: str
    template: str


@router.get("", response_model=List[PromptEntry])
def list_prompts() -> List[PromptEntry]:
    """Return all prompt entries with metadata — used by the PromptEditor UI."""
    try:
        registry = PromptRegistry.instance()
        entries = []
        for item in registry.list_all():
            entries.append(PromptEntry(
                key=item["key"],
                method_name=item.get("method_name", item["key"]),
                method_id=item.get("method_id", 0),
                pipeline_stage=item.get("pipeline_stage", ""),
                description=item.get("description", "").strip(),
                template=item.get("template", ""),
            ))
        # Sort by method_id so UI displays in pipeline order
        entries.sort(key=lambda e: (e.method_id, e.key))
        return entries
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/{key}", response_model=Dict[str, str])
def update_prompt(key: str, req: PromptUpdateRequest) -> Dict[str, str]:
    """Update a prompt template. Change takes effect on the next search request."""
    try:
        PromptRegistry.instance().update_prompt(key, req.template)
        return {"status": "updated", "key": key}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {key!r}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{key}/reset", response_model=Dict[str, str])
def reset_prompt(key: str) -> Dict[str, str]:
    """Reset a prompt to its factory default (as defined in the original prompts.yaml)."""
    try:
        default = PromptRegistry.instance().reset_prompt(key)
        return {"status": "reset", "key": key, "template": default}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {key!r}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
