"""
ChunkerBase — abstract interface all chunkers implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.chunking.models import Chunk


class ChunkerBase(ABC):
    """
    All chunkers implement this interface.
    Input: raw document text + doc metadata.
    Output: list of Chunk objects with provenance fields populated.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> List[Chunk]:
        """
        Split text into chunks.

        Parameters
        ----------
        text     : Full document text.
        doc_id   : Parent document identifier.
        metadata : Arbitrary metadata copied to every chunk.

        Returns
        -------
        List of Chunk objects, ordered as they appear in the source.
        """
