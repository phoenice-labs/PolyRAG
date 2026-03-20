"""
Root conftest — shared fixtures for all phases.
Downloads and caches Shakespeare from Project Gutenberg.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


@pytest.fixture(scope="session")
def shakespeare_raw() -> str:
    """
    Full raw text of Project Gutenberg's Complete Works of Shakespeare.
    Downloaded once per test session and cached to data/shakespeare.txt.
    """
    if not SHAKESPEARE_PATH.exists():
        import requests

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[conftest] Downloading Shakespeare from {GUTENBERG_URL} ...")
        r = requests.get(GUTENBERG_URL, timeout=120)
        r.raise_for_status()
        SHAKESPEARE_PATH.write_text(r.text, encoding="utf-8")
        print(f"[conftest] Saved {len(r.text):,} chars to {SHAKESPEARE_PATH}")

    return SHAKESPEARE_PATH.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="session")
def shakespeare_clean(shakespeare_raw: str) -> str:
    """Shakespeare text with Gutenberg header/footer stripped."""
    from core.ingestion.loader import strip_gutenberg_header_footer

    return strip_gutenberg_header_footer(shakespeare_raw)
