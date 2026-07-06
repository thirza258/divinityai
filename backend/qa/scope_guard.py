"""
Scope guard — reject out-of-domain queries and low-confidence intents.

Pure Python, no LLM call.
"""

OFF_DOMAIN_MESSAGE = (
    "This system only answers questions grounded in the Quran and "
    "authenticated Hadith collections. Your question appears to be "
    "outside this scope. Please rephrase with a specific Islamic topic."
)

LOW_CONFIDENCE_MESSAGE = (
    "I'm not confident I can answer this question from the Quran and "
    "Hadith sources available. Please rephrase your question."
)


def check_scope(intent: str, confidence: float) -> dict:
    """Check if a query is within scope.

    Returns a dict with ``allowed`` (bool) and ``message`` (str).
    """
    if intent == 'off_domain':
        return {'allowed': False, 'message': OFF_DOMAIN_MESSAGE}

    if confidence < 0.6:
        return {'allowed': False, 'message': LOW_CONFIDENCE_MESSAGE}

    return {'allowed': True, 'message': ''}