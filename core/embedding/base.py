"""
Embedding provider abstraction.
All providers return List[List[float]] for a batch of texts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProviderBase(ABC):
    """Common interface for all embedding providers."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Parameters
        ----------
        texts : List of strings to embed (must be non-empty).

        Returns
        -------
        List of float vectors, same length as input.
        """

    def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper: embed a single string."""
        return self.embed([text])[0]
