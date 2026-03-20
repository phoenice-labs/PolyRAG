"""
Chunks router: POST /api/chunks/preview
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException

from api.schemas import ChunkItem, ChunkPreviewRequest, ChunkPreviewResponse

router = APIRouter(tags=["chunks"])


def _run_chunk_preview(
    text: str, strategy: str, chunk_size: int, overlap: int
) -> ChunkPreviewResponse:
    """Run chunking pipeline without ingestion."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from core.chunking.section_aware import SectionAwareChunker
        from core.chunking.sentence_boundary import SentenceBoundaryChunker
        from core.chunking.fixed_overlap import FixedOverlapChunker
        from core.chunking.pipeline import ChunkingPipeline
        from core.chunking.models import ChunkRegistry

        if strategy == "sentence":
            chunker = SentenceBoundaryChunker(max_words=max(chunk_size // 5, 20), overlap_sents=1)
        elif strategy in ("sliding", "paragraph"):
            chunker = FixedOverlapChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        else:
            chunker = SectionAwareChunker(child_size=chunk_size, child_overlap=overlap)

        registry = ChunkRegistry()
        pipeline = ChunkingPipeline(chunker=chunker, registry=registry)

        chunks = pipeline.run(text, doc_id="preview_doc", metadata={"source": "preview"})

        items: List[ChunkItem] = []
        for i, chunk in enumerate(chunks):
            items.append(
                ChunkItem(
                    index=i,
                    text=chunk.text,
                    tokens=chunk.token_count,
                    char_start=chunk.start_char,
                    char_end=chunk.end_char,
                    parent_id=chunk.parent_id,
                    chunk_type=chunk.chunk_type,
                    entities=[],
                )
            )

        total_chars = sum(len(c.text) for c in items)
        avg_size = total_chars / max(len(items), 1)

        return ChunkPreviewResponse(
            chunks=items,
            total_chunks=len(items),
            total_chars=total_chars,
            avg_chunk_size=round(avg_size, 2),
            strategy=strategy,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chunks/preview", response_model=ChunkPreviewResponse)
async def preview_chunks(req: ChunkPreviewRequest) -> ChunkPreviewResponse:
    """Dry-run chunking on input text/file without ingesting."""
    text = req.text
    if not text and req.corpus_path:
        try:
            with open(req.corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read corpus_path: {exc}")

    if not text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'corpus_path'")

    return await asyncio.to_thread(
        _run_chunk_preview, text, req.strategy, req.chunk_size, req.overlap
    )
