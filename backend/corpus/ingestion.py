"""
Corpus loading — reads JSON files and returns structured records
for ingestion into ChromaDB and BM25 indexes.
"""

import json
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def load_quran_corpus(data_dir: Path) -> list[dict]:
    """Load Quran ayahs from JSON files in *data_dir*.

    Expects a single ``quran_ayahs.json`` file with combined Arabic
    and English text.  Each ayah is one chunk.
    """
    path = data_dir / "quran_ayahs.json"
    if not path.is_file():
        raise FileNotFoundError(f"Quran corpus file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        ayahs = json.load(f)

    logger.info("Loaded %d Quran ayahs from %s", len(ayahs), path)
    return ayahs


def load_hadith_corpus(data_dir: Path, collections: list[str]) -> list[dict]:
    """Load Hadith collections from JSON files in *data_dir*.

    Expects files named ``hadith_{collection}.json`` for each
    collection in *collections*.
    """
    all_hadith: list[dict] = []

    for collection in collections:
        path = data_dir / f"hadith_{collection}.json"
        if not path.is_file():
            logger.warning("Hadith corpus file not found, skipping: %s", path)
            continue

        with open(path, 'r', encoding='utf-8') as f:
            hadiths = json.load(f)

        all_hadith.extend(hadiths)
        logger.info("Loaded %d hadiths from %s", len(hadiths), path)

    logger.info("Loaded %d total hadiths from %d collections", len(all_hadith), len(collections))
    return all_hadith