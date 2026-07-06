"""
API views for the Islamic RAG query pipeline.
"""

import logging

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import QueryRequestSerializer, QueryResponseSerializer
from .pipeline import PipelineService

logger = logging.getLogger(__name__)

PHASE = getattr(settings, 'RAG_PHASE', 1)


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
        except Exception as exc:
            logger.exception("Pipeline failed for query: %s", data['query'][:100])
            return Response(
                {'error': 'Pipeline processing failed', 'detail': str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_serializer = QueryResponseSerializer(data=result)
        if response_serializer.is_valid():
            return Response(response_serializer.validated_data)
        # Fallback: return raw result if serialization fails
        return Response(result)


class HealthView(APIView):
    """GET /api/v1/health — Health check."""

    def get(self, request):
        return Response({'status': 'ok', 'phase': PHASE})


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