"""
Prompt Registry — loads prompts from config/prompts.yaml at startup and caches them.

Usage
-----
    from core.prompt_registry import PromptRegistry

    # Get the system prompt for a specific method
    registry = PromptRegistry.instance()
    system = registry.get_prompt("query_rewriter")

    # Update a prompt (persists to YAML)
    registry.update_prompt("query_rewriter", "New prompt text...")

    # Reset to factory default
    registry.reset_prompt("query_rewriter")

    # List all prompts with metadata
    all_prompts = registry.list_all()

The registry is a module-level singleton so all pipeline components share the same
in-memory prompts. When a prompt is updated via the UI, the change is:
  1. Applied immediately in memory (next search picks it up)
  2. Persisted to config/prompts.yaml (survives server restarts)
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default path, relative to project root
_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


class PromptRegistry:
    """
    Thread-safe singleton registry for all LLM prompts.

    Prompts are loaded from config/prompts.yaml and kept in memory.
    Updates are written back to the YAML file immediately.
    """

    _instance: Optional["PromptRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, yaml_path: Path = _DEFAULT_YAML) -> None:
        self._path = yaml_path
        self._prompts: Dict[str, Dict[str, Any]] = {}
        self._defaults: Dict[str, Dict[str, Any]] = {}
        self._file_lock = threading.Lock()
        self._load()

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls, yaml_path: Path = _DEFAULT_YAML) -> "PromptRegistry":
        """Return the module-level singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(yaml_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_prompt(self, key: str) -> str:
        """Return the current template for the given prompt key."""
        entry = self._prompts.get(key)
        if entry is None:
            raise KeyError(f"Unknown prompt key: {key!r}. Available: {list(self._prompts)}")
        return entry.get("template", "")

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Return metadata (method_name, method_id, pipeline_stage, description) without template."""
        entry = self._prompts.get(key, {})
        return {k: v for k, v in entry.items() if k != "template"}

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all prompts as a list of dicts (including key + metadata + template)."""
        result = []
        for key, entry in self._prompts.items():
            result.append({"key": key, **entry})
        return result

    def update_prompt(self, key: str, template: str) -> None:
        """
        Update a prompt template in memory and persist to YAML.

        The change takes effect immediately — the next search will use the new prompt.
        """
        if key not in self._prompts:
            raise KeyError(f"Unknown prompt key: {key!r}")
        with self._file_lock:
            self._prompts[key]["template"] = template
            self._save()
        logger.info("PromptRegistry: updated prompt %r", key)

    def reset_prompt(self, key: str) -> str:
        """
        Reset a prompt to its factory default (from the original YAML on disk).

        Returns the default template string.
        """
        if key not in self._defaults:
            raise KeyError(f"Unknown prompt key: {key!r}")
        default_template = self._defaults[key].get("template", "")
        self.update_prompt(key, default_template)
        logger.info("PromptRegistry: reset prompt %r to default", key)
        return default_template

    def keys(self) -> List[str]:
        return list(self._prompts.keys())

    # ── Private ────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load prompts from YAML file. Falls back to empty registry on error."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw = data.get("prompts", {})
            self._prompts = {k: dict(v) for k, v in raw.items()}
            # Store a deep copy as the immutable defaults for reset operations
            self._defaults = copy.deepcopy(self._prompts)
            logger.debug("PromptRegistry: loaded %d prompts from %s", len(self._prompts), self._path)
        except FileNotFoundError:
            logger.warning("PromptRegistry: %s not found — using empty registry", self._path)
            self._prompts = {}
            self._defaults = {}
        except Exception as exc:
            logger.warning("PromptRegistry: failed to load %s — %s", self._path, exc)
            self._prompts = {}
            self._defaults = {}

    def _save(self) -> None:
        """Persist current prompts to YAML file."""
        try:
            data = {"prompts": self._prompts}
            with open(self._path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            logger.error("PromptRegistry: failed to save %s — %s", self._path, exc)
