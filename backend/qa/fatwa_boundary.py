"""
Fatwa boundary trigger — rule-based keyword detection.

Scans the generated answer for sensitive Islamic jurisprudence topics
and appends a disclaimer when triggered.  Pure Python, no LLM call.
"""

SENSITIVE_KEYWORDS = [
    # Divorce
    'طلاق', 'talaq', 'divorce', 'khula', 'خلع',
    # Inheritance
    'ميراث', 'inheritance', 'faraid', 'فرائض',
    # Finance / usury
    'ربا', 'interest', 'usury', 'مصرف', 'banking',
    # Medical
    'إجهاض', 'abortion', 'تشريح', 'autopsy',
    # Political
    'جهاد', 'jihad', 'خلافة', 'caliphate',
]

DISCLAIMER = (
    "Note: This answer involves sensitive Islamic jurisprudence. "
    "For a definitive ruling, please consult a qualified scholar. "
    "The above is a presentation of source texts, not a fatwa."
)


def check_fatwa_boundary(answer: str) -> dict:
    """Check if the answer triggers the fatwa boundary.

    Returns a dict with ``triggered`` (bool) and ``disclaimer`` (str or None).
    """
    if not answer:
        return {'triggered': False, 'disclaimer': None}

    answer_lower = answer.lower()

    for keyword in SENSITIVE_KEYWORDS:
        if keyword.lower() in answer_lower or keyword in answer:
            return {'triggered': True, 'disclaimer': DISCLAIMER}

    return {'triggered': False, 'disclaimer': None}