"""
Ground truth dataset registry — versioned evaluation corpora.

Provides persistent, named storage for Q&A evaluation datasets so teams can:
  - Reuse the same corpus across multiple evaluation runs
  - Track dataset evolution (version auto-increments on every save)
  - Trigger full eval runs against a stored dataset via the API

Storage layout::

    data/eval_datasets/
        {name}.json   ← {"meta": {...}, "items": [...]}

Usage::

    registry = DatasetRegistry()

    meta = registry.save(
        "hamlet-qa",
        [EvalDatasetItem("Who is Hamlet?", "A Danish prince", ["hamlet.txt"])],
        description="Core Hamlet Q&A — v1",
    )
    print(meta.version)       # 1

    dataset = registry.load("hamlet-qa")
    print(dataset.items[0].question)

    meta2 = registry.save("hamlet-qa", updated_items, description="Added act-II questions")
    print(meta2.version)      # 2  (auto-incremented)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DIR = Path(__file__).parent.parent.parent / "data" / "eval_datasets"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


# ── Domain objects ─────────────────────────────────────────────────────────────

@dataclass
class EvalDatasetItem:
    """One evaluation question with expected answer and optional source hints."""
    question: str
    expected_answer: str
    expected_sources: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_sources": self.expected_sources,
        }


@dataclass
class EvalDatasetMeta:
    """Metadata header stored alongside every evaluation dataset."""
    name: str
    version: int = 1
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    item_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalDataset:
    """A versioned evaluation dataset (metadata + question items)."""
    meta: EvalDatasetMeta
    items: List[EvalDatasetItem] = field(default_factory=list)

    def to_eval_questions(self) -> List[Dict[str, Any]]:
        """
        Return items formatted for the ``EvaluateRequest.questions`` schema.

        Each dict has ``question``, ``expected_answer``, and ``expected_sources``.
        """
        return [it.as_dict() for it in self.items]


# ── Registry ───────────────────────────────────────────────────────────────────

class DatasetRegistry:
    """
    Persistent, versioned storage for named evaluation datasets.

    Parameters
    ----------
    base_dir :
        Directory where dataset JSON files are stored.
        Defaults to ``<repo_root>/data/eval_datasets/``.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._dir = Path(base_dir) if base_dir else _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Mutating operations ────────────────────────────────────────────────────

    def save(
        self,
        name: str,
        items: List[EvalDatasetItem],
        description: str = "",
    ) -> EvalDatasetMeta:
        """
        Save (or update) a named dataset.

        If a dataset with *name* already exists its version is incremented and
        ``updated_at`` is refreshed; all items are replaced with the new list.

        Parameters
        ----------
        name        : Dataset identifier — lowercase alphanumeric + hyphens/underscores.
        items       : Ordered list of ``EvalDatasetItem``.
        description : Human-readable annotation (kept from previous save if omitted).

        Returns
        -------
        EvalDatasetMeta — metadata of the persisted dataset.
        """
        _validate_name(name)
        now = _utc_now()
        path = self._path(name)

        if path.exists():
            existing_raw = self._load_raw(name)
            version = existing_raw["meta"].get("version", 1) + 1
            created_at = existing_raw["meta"].get("created_at", now)
            # keep existing description when caller passes empty string
            resolved_desc = description if description else existing_raw["meta"].get("description", "")
        else:
            version = 1
            created_at = now
            resolved_desc = description

        meta = EvalDatasetMeta(
            name=name,
            version=version,
            description=resolved_desc,
            created_at=created_at,
            updated_at=now,
            item_count=len(items),
        )

        payload: Dict[str, Any] = {
            "meta": meta.as_dict(),
            "items": [it.as_dict() for it in items],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    def delete(self, name: str) -> bool:
        """
        Delete a dataset by name.

        Returns True if the dataset was found and removed, False if it did not exist.
        """
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Query operations ───────────────────────────────────────────────────────

    def load(self, name: str) -> EvalDataset:
        """
        Load a dataset by name.

        Raises
        ------
        KeyError if the dataset does not exist.
        """
        raw = self._load_raw(name)
        m = raw["meta"]
        meta = EvalDatasetMeta(
            name=m.get("name", name),
            version=m.get("version", 1),
            description=m.get("description", ""),
            created_at=m.get("created_at", ""),
            updated_at=m.get("updated_at", ""),
            item_count=m.get("item_count", 0),
        )
        items = [
            EvalDatasetItem(
                question=it["question"],
                expected_answer=it.get("expected_answer", ""),
                expected_sources=it.get("expected_sources", []),
            )
            for it in raw.get("items", [])
        ]
        return EvalDataset(meta=meta, items=items)

    def list_datasets(self) -> List[EvalDatasetMeta]:
        """Return metadata for every stored dataset, sorted alphabetically by name."""
        results: List[EvalDatasetMeta] = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                m = raw.get("meta", {})
                results.append(EvalDatasetMeta(
                    name=m.get("name", p.stem),
                    version=m.get("version", 1),
                    description=m.get("description", ""),
                    created_at=m.get("created_at", ""),
                    updated_at=m.get("updated_at", ""),
                    item_count=m.get("item_count", 0),
                ))
            except Exception:
                continue
        return results

    def exists(self, name: str) -> bool:
        """Return True if a dataset with the given name exists."""
        return self._path(name).exists()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    def _load_raw(self, name: str) -> Dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"Dataset '{name}' not found in registry")
        return json.loads(path.read_text(encoding="utf-8"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_name(name: str) -> None:
    """Raise ValueError if *name* is not a valid dataset identifier."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid dataset name '{name}'. "
            "Use 1–64 characters: lowercase letters, digits, hyphens, or "
            "underscores — must start with a letter or digit."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: Optional[DatasetRegistry] = None


def get_dataset_registry() -> DatasetRegistry:
    """Return the module-level DatasetRegistry singleton (lazy-initialised)."""
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
