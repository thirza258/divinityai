"""
Arabic text normalization for BM25 indexing and citation verification.

Normalization steps (in order):
1. Unicode NFKD normalization — decompose precomposed characters
2. Strip tashkeel (diacritics: fatha, damma, kasra, sukun, shadda, etc.)
3. Normalize alef variants (أ, إ, آ → ا)
4. Strip tatweel (kashida: ـ)
5. Collapse whitespace

The same function is used at ingestion time, query time, and during
citation verification to ensure consistent text comparison.
"""

import re
import unicodedata

# Unicode ranges for Arabic diacritics (tashkeel)
# U+064B..U+065F: fatha, damma, kasra, fathatan, dammatan, kasratan, shadda, sukun, etc.
# U+0670: superscript alef (dagger alef)
TASHKEEL_PATTERN = re.compile(r'[ً-ٰٟ]')

# Tatweel / kashida character
TATWEEL_PATTERN = re.compile(r'ـ+')

# Alef variants → bare alef
ALEF_MAP = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا'})


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison and BM25 tokenization.

    Does NOT strip original alef at end of words (ى → ي) or
    handle hamza-on-waw (ؤ) — those are kept to maintain
    word-boundary distinctions important for retrieval.

    Returns the normalized string.
    """
    if not text:
        return ""

    # Step 1: NFKD normalization — decomposes e.g. أ (precomposed) → ا + ◌ٔ
    text = unicodedata.normalize('NFKD', text)

    # Step 2: Strip tashkeel / diacritics
    text = TASHKEEL_PATTERN.sub('', text)

    # Step 3: Normalize alef variants
    text = text.translate(ALEF_MAP)

    # Step 4: Strip tatweel
    text = TATWEEL_PATTERN.sub('', text)

    # Step 5: Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text