"""
API views for the Islamic RAG query pipeline.
"""

import logging
import time

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import QueryRequestSerializer, QueryResponseSerializer
from .pipeline import PipelineService

logger = logging.getLogger(__name__)

PHASE = getattr(settings, 'RAG_PHASE', 1)


# ---------------------------------------------------------------------------
# Health-check helpers
# ---------------------------------------------------------------------------

def _check_ollama() -> dict:
    """Check Ollama is reachable and can produce embeddings."""
    start = time.monotonic()
    OLLAMA_EMBED_MODEL = "unknown"
    try:
        from retrieval.dense_rag import _embed_single, OLLAMA_EMBED_MODEL

        embedding = _embed_single("test")
        latency_ms = round((time.monotonic() - start) * 1000)
        if embedding and len(embedding) > 0:
            return {
                "status": "ok",
                "model": OLLAMA_EMBED_MODEL,
                "latency_ms": latency_ms,
            }
        return {
            "status": "degraded",
            "model": OLLAMA_EMBED_MODEL,
            "latency_ms": latency_ms,
            "detail": "Returned empty embedding",
        }
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000)
        return {
            "status": "error",
            "model": OLLAMA_EMBED_MODEL,
            "latency_ms": latency_ms,
            "detail": str(exc),
        }


def _check_chroma() -> dict:
    """Check ChromaDB is reachable and collections exist."""
    start = time.monotonic()
    host = "unknown"
    try:
        from chroma.chroma_utils import get_chroma_client, get_or_create_collection
        from chroma.chroma_settings import CHROMA_HOST, CHROMA_PORT
        from retrieval.dense_rag import QURAN_COLLECTION, HADITH_COLLECTION

        host = f"{CHROMA_HOST}:{CHROMA_PORT}"
        client = get_chroma_client()
        collections_info = {}
        for name in [QURAN_COLLECTION, HADITH_COLLECTION]:
            try:
                col = get_or_create_collection(name=name)
                collections_info[name] = col.count()
            except Exception:
                collections_info[name] = -1

        latency_ms = round((time.monotonic() - start) * 1000)

        # Determine status: ok if all have docs, degraded if some missing
        counts = collections_info.values()
        if all(c > 0 for c in counts):
            overall = "ok"
        elif any(c > 0 for c in counts):
            overall = "degraded"
        elif all(c == 0 for c in counts):
            overall = "degraded"
        else:
            overall = "error"

        return {
            "status": overall,
            "host": host,
            "latency_ms": latency_ms,
            "collections": collections_info,
        }
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000)
        return {
            "status": "error",
            "host": host,
            "latency_ms": latency_ms,
            "detail": str(exc),
        }


PIPELINE_ERROR_MESSAGE = (
    "Sorry — something went wrong while processing your question. "
    "Please try again in a moment."
)


def _pipeline_error_response(query: str) -> dict:
    """Chat-shaped body returned when the pipeline fails.

    Errors are delivered as a normal chat answer (HTTP 200) so the client
    always renders a readable message instead of an HTTP error page.
    """
    return {
        'query': query,
        'intent': 'error',
        'answer': PIPELINE_ERROR_MESSAGE,
        'sources': [],
        'citations': [],
        'safety': {
            'hallucination_detected': False,
            'flagged_spans': [],
            'fatwa_boundary_triggered': False,
            'disclaimer': None,
        },
        'pipeline_meta': {'phase': PHASE, 'llm_calls': 0, 'retrieval_iterations': 0},
        'error': True,
    }


class QueryView(APIView):
    """POST /api/v1/query — Run the full Islamic RAG pipeline."""

    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        try:
            pipeline = PipelineService(phase=PHASE)
            result = pipeline.run(
                query=data['query'],
                language=data['language'],
                max_sources=data['max_sources'],
            )
        except Exception:
            logger.exception("Pipeline failed for query: %s", data['query'][:100])
            return Response(_pipeline_error_response(data['query']))

        response_serializer = QueryResponseSerializer(data=result)
        if response_serializer.is_valid():
            return Response(response_serializer.validated_data)
        # Fallback: return raw result if serialization fails
        logger.warning("Response serialization failed: %s", response_serializer.errors)
        return Response(result)


class HealthView(APIView):
    """GET /api/v1/health — Health check with service verification.

    Checks:
    - Ollama embedding service (reachable, returns valid embeddings)
    - ChromaDB (reachable, collections have documents)
    - API itself (always ok if this endpoint responds)
    """

    def get(self, request):
        checks = {
            "api": {"status": "ok"},
            "ollama": _check_ollama(),
            "chroma": _check_chroma(),
        }

        # Overall status is the worst of all checks
        statuses = {c["status"] for c in checks.values()}
        if "error" in statuses:
            overall = "error"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "ok"

        return Response({
            "status": overall,
            "phase": PHASE,
            "checks": checks,
        })


class CorpusStatsView(APIView):
    """GET /api/v1/corpus/stats — Corpus statistics."""

    def get(self, request):
        from chroma.chroma_utils import get_or_create_collection
        from retrieval.dense_rag import QURAN_COLLECTION, HADITH_COLLECTION

        stats = {}
        for name in [QURAN_COLLECTION, HADITH_COLLECTION]:
            try:
                collection = get_or_create_collection(name=name)
                stats[name] = {'document_count': collection.count()}
            except Exception:
                stats[name] = {'document_count': 0}
        return Response(stats)