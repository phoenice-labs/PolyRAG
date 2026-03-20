"""
SentenceTransformer embedding provider (fully open-source, runs locally).

Supported models (all Apache 2.0, all run on CPU):

  Model                    Dim   Size    Speed (CPU)   MTEB Score
  ─────────────────────────────────────────────────────────────────
  all-MiniLM-L6-v2         384   ~80 MB  ~5 ms/chunk   56.3  (default — fast, low-RAM)
  BAAI/bge-base-en-v1.5    768   ~440 MB ~15 ms/chunk  63.6  (balanced quality/speed)
  BAAI/bge-large-en-v1.5  1024   ~1.3 GB ~40 ms/chunk  64.2  (best quality, slow)

Each model uses its own isolated vector collection (suffix: minilm / bge-base / bge-large)
so that different-dimension vectors never mix. Switching model requires re-ingestion.

BGE v1.5 note: query instructions are NOT required (removed in v1.5 vs v1).

Requires: pip install sentence-transformers
"""
from __future__ import annotations

import threading
from typing import List

from core.embedding.base import EmbeddingProviderBase

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Supported models with expected dimensions (for documentation and validation)
SUPPORTED_MODELS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

# Module-level cache: (model_name, device) → (SentenceTransformer, dim)
# Shared across all provider instances to avoid concurrent loading issues.
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}
_MODEL_LOCK = threading.Lock()


class SentenceTransformerProvider(EmbeddingProviderBase):
    """
    Local sentence-transformers embedding provider.

    Config keys
    -----------
    model      : HuggingFace model name (default: all-MiniLM-L6-v2)
    device     : "cpu" (default) | "cuda" | "mps"
    batch_size : embedding batch size (default: 32)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._model = None
        self._dim: int | None = None

    def _load(self) -> None:
        model_name = self.config.get("model", _DEFAULT_MODEL)
        device = self.config.get("device", "cpu")
        cache_key = (model_name, device)

        # Fast path: already cached
        if cache_key in _MODEL_CACHE:
            self._model, self._dim = _MODEL_CACHE[cache_key]
            return

        # Serialize concurrent model loading — PyTorch init is not thread-safe
        with _MODEL_LOCK:
            # Re-check inside lock in case another thread just loaded it
            if cache_key in _MODEL_CACHE:
                self._model, self._dim = _MODEL_CACHE[cache_key]
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                ) from e
            model = SentenceTransformer(model_name, device=device)
            dim = model.get_sentence_embedding_dimension()
            _MODEL_CACHE[cache_key] = (model, dim)
            self._model, self._dim = model, dim

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        batch_size = self.config.get("batch_size", 32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalised for cosine similarity
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


class EmbeddingRegistry:
    """Factory for embedding providers."""

    _REGISTRY = {
        "sentence_transformer": SentenceTransformerProvider,
    }

    @classmethod
    def create(cls, provider: str, config: dict) -> EmbeddingProviderBase:
        key = provider.lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Available: {sorted(cls._REGISTRY.keys())}"
            )
        return cls._REGISTRY[key](config)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._REGISTRY.keys())
