"""
Document loaders — load raw text from local files or remote URLs.
Includes a helper to download and cache Project Gutenberg texts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


GUTENBERG_SHAKESPEARE_URL = (
    "https://www.gutenberg.org/cache/epub/100/pg100.txt"
)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ── Generic loaders ───────────────────────────────────────────────────────────

def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a local text file and return its contents."""
    return Path(path).read_text(encoding=encoding, errors="replace")


def load_from_url(url: str, timeout: int = 60) -> str:
    """Download text from a URL and return its contents."""
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


# ── Project Gutenberg helpers ─────────────────────────────────────────────────

def download_gutenberg(
    url: str = GUTENBERG_SHAKESPEARE_URL,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> str:
    """
    Download a Project Gutenberg text and cache it locally.

    Parameters
    ----------
    url        : Gutenberg plain-text URL.
    cache_path : Local path to cache the file (default: data/shakespeare.txt).
    force      : Re-download even if cache exists.

    Returns
    -------
    Full text of the downloaded file.
    """
    if cache_path is None:
        cache_path = _DATA_DIR / "shakespeare.txt"

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists() or force:
        print(f"Downloading {url} → {cache_path} ...")
        text = load_from_url(url)
        cache_path.write_text(text, encoding="utf-8")
        print(f"  Saved {len(text):,} characters.")
    else:
        text = cache_path.read_text(encoding="utf-8", errors="replace")

    return text


def strip_gutenberg_header_footer(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate header and footer.
    The actual content sits between START/END markers.
    """
    start_pattern = re.compile(
        r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
    )
    end_pattern = re.compile(
        r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
    )

    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    if start_match:
        text = text[start_match.end():]
        # Skip the rest of the header line
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text

    if end_match:
        # Search in original; adjust offset
        abs_end = end_pattern.search(text)
        if abs_end:
            text = text[: abs_end.start()]

    return text.strip()


def naive_chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separator: str = "\n\n",
) -> list[str]:
    """
    Simple fixed-size chunker used in Phase 1 ingestion.
    Phase 2 replaces this with the full semantic chunking pipeline.

    Parameters
    ----------
    text       : Raw text to chunk.
    chunk_size : Target chunk size in characters.
    overlap    : Overlap between consecutive chunks in characters.
    separator  : Preferred split boundary (paragraph break by default).

    Returns
    -------
    List of non-empty text chunks.
    """
    # Split on preferred separator first
    paragraphs = [p.strip() for p in text.split(separator) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Handle very long single paragraphs
            if len(para) > chunk_size:
                for start in range(0, len(para), chunk_size - overlap):
                    sub = para[start: start + chunk_size]
                    if sub.strip():
                        chunks.append(sub.strip())
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks
