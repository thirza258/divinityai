"""
Scope guard — reject out-of-domain queries.

Pure Python, no LLM call.
"""

OFF_DOMAIN_MESSAGE = (
    "This system only answers questions grounded in the Quran and "
    "authenticated Hadith collections. Your question appears to be "
    "outside this scope. Please rephrase with a specific Islamic topic."
)

# Only a classifier that is reasonably sure a query is off-domain may block
# it.  Below this, the query goes to retrieval instead: grounded generation
# is itself a scope filter, and refusing an in-domain question the router
# merely mis-scored is the worse failure.
OFF_DOMAIN_MIN_CONFIDENCE = 0.5


def check_scope(intent: str, confidence: float) -> dict:
    """Check if a query is within scope.

    Returns a dict with ``allowed`` (bool) and ``message`` (str).
    """
    if intent == 'off_domain' and confidence >= OFF_DOMAIN_MIN_CONFIDENCE:
        return {'allowed': False, 'message': OFF_DOMAIN_MESSAGE}

    return {'allowed': True, 'message': ''}
