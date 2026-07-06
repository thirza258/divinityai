"""
URL configuration for the DivinityAI backend.
"""

from django.contrib import admin
from django.urls import path

from router.views import (
    chroma_collection_detail,
    chroma_collections,
    chroma_documents,
    chroma_query,
    llm_generate,
)
from qa.views import CorpusStatsView, HealthView, QueryView

urlpatterns = [
    path("admin/", admin.site.urls),

    # ------------------------------------------------------------------
    # ChromaDB — Collections
    # ------------------------------------------------------------------
    path(
        "api/chroma/collections/",
        chroma_collections,
        name="chroma-collections",
    ),
    path(
        "api/chroma/collections/<str:name>/",
        chroma_collection_detail,
        name="chroma-collection-detail",
    ),

    # ------------------------------------------------------------------
    # ChromaDB — Documents
    # ------------------------------------------------------------------
    path(
        "api/chroma/collections/<str:collection_name>/documents/",
        chroma_documents,
        name="chroma-documents",
    ),

    # ------------------------------------------------------------------
    # ChromaDB — Query
    # ------------------------------------------------------------------
    path(
        "api/chroma/collections/<str:collection_name>/query/",
        chroma_query,
        name="chroma-query",
    ),

    # ------------------------------------------------------------------
    # LLM — Generate
    # ------------------------------------------------------------------
    path(
        "api/generate/",
        llm_generate,
        name="llm-generate",
    ),

    # ------------------------------------------------------------------
    # RAG — Islamic QA Pipeline
    # ------------------------------------------------------------------
    path(
        "api/v1/query",
        QueryView.as_view(),
        name="qa-query",
    ),
    path(
        "api/v1/health",
        HealthView.as_view(),
        name="qa-health",
    ),
    path(
        "api/v1/corpus/stats",
        CorpusStatsView.as_view(),
        name="qa-corpus-stats",
    ),
]
