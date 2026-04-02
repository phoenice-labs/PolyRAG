"""
Document loaders — load raw text from local files or remote URLs.
Includes a helper to download and cache Project Gutenberg texts.

New in this version
-------------------
``load_document()`` is the recommended entry-point for all file-based
ingestion.  It auto-detects the file format (plain text, PDF, PPTX) and
delegates to the appropriate extractor in ``core.ingestion.extractors``.

``load_text_file()`` is kept **unchanged** for backward compatibility.
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


def load_document(path: str | Path, enable_rich_formats: bool = True) -> str:
    """
    Load any supported document format and return its plain-text content.

    Supported formats
    -----------------
    - Plain text / Markdown / CSV / JSON / HTML  (always available)
    - PDF                                         (requires ``pypdf>=4.0``)
    - PowerPoint .pptx                            (requires ``python-pptx>=1.0``)

    Parameters
    ----------
    path               : Path to the document.
    enable_rich_formats: When ``False``, only plain-text files are accepted.
                         Set ``ingestion.enable_rich_formats: false`` in
                         config.yaml to toggle this at deployment time.

    Returns
    -------
    Plain-text string ready for chunking.
    """
    from core.ingestion.extractors import extract_text
    return extract_text(Path(path), enable_rich_formats=enable_rich_formats)


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


# ── Streaming chunker (large document support) ────────────────────────────────

def stream_chunk_file(
    path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
    encoding: str = "utf-8",
    max_doc_size_mb: float = 200.0,
):
    """
    Stream-chunk a large text file without loading it fully into RAM.

    Yields one chunk (str) at a time. Safe for files of arbitrary size — only
    a sliding window of `chunk_size + overlap` characters is held in memory
    at any point.

    Parameters
    ----------
    path           : Path to the text file.
    chunk_size     : Target chunk size in characters.
    overlap        : Character overlap between consecutive chunks.
    encoding       : File encoding (default utf-8, errors replaced).
    max_doc_size_mb: Guard rail — raises ValueError if file exceeds this size.
                     Set to 0 to disable the check.

    Yields
    ------
    str — individual text chunks, each ≤ chunk_size characters (approximately).

    Usage
    -----
    for chunk in stream_chunk_file("large.txt", chunk_size=512, overlap=64):
        process(chunk)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if max_doc_size_mb > 0 and file_size_mb > max_doc_size_mb:
        raise ValueError(
            f"File '{path.name}' is {file_size_mb:.1f} MB, which exceeds "
            f"max_doc_size_mb={max_doc_size_mb}. "
            f"Split the file or increase max_doc_size_mb in the profile's scale_hints."
        )

    buffer = ""
    read_size = chunk_size * 8  # read in larger blocks for I/O efficiency

    with path.open(encoding=encoding, errors="replace") as f:
        while True:
            block = f.read(read_size)
            if not block:
                break
            buffer += block

            # Emit full chunks from the buffer, retaining the overlap tail
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size].strip()
                if chunk:
                    yield chunk
                # Slide forward, keeping the overlap for context continuity
                buffer = buffer[chunk_size - overlap:]

    # Emit the remaining tail
    tail = buffer.strip()
    if tail:
        yield tail


def estimate_chunk_count(path: str | Path, chunk_size: int = 512, overlap: int = 64) -> int:
    """
    Fast estimate of total chunks for a file — used for progress reporting.
    Does not read the file; estimates from file size.

    Returns 0 if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    file_size_bytes = path.stat().st_size
    effective_step = max(1, chunk_size - overlap)
    return max(1, int(file_size_bytes / (effective_step * 2.5)))
