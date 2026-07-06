"""
Router API views — ChromaDB collection management and LLM text generation.
"""

import json
import logging

from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from chroma.chroma_settings import CHROMA_DEFAULT_COLLECTION
from chroma.chroma_utils import (
    add_documents,
    delete_collection,
    delete_documents,
    get_collection_count,
    get_documents,
    get_or_create_collection,
    list_collections,
    query_documents,
    update_documents,
)
from generation.llm_service import generate

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================

def _parse_json_body(request: HttpRequest) -> dict:
    """Parse the JSON body of a request, returning a dict.

    Returns an empty dict when the body is empty or not valid JSON.
    """
    try:
        body = request.body.decode("utf-8")
        if not body.strip():
            return {}
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to parse request body: %s", exc)
        return {}


def _error(message: str, status: int = 400) -> JsonResponse:
    """Return a standardised JSON error response."""
    return JsonResponse({"error": message}, status=status)


# ======================================================================
# ChromaDB — Collections
# ======================================================================

@csrf_exempt
def chroma_collections(request: HttpRequest) -> JsonResponse:
    """Manage ChromaDB collections.

    ``GET``
        List all collection names.

    ``POST``
        Create (or get) a collection.

        Body (JSON)::

            {"name": "my_collection",   // optional — defaults to configured default
             "metadata": {"hnsw:space": "cosine"}}  // optional
    """
    if request.method == "GET":
        try:
            names = list_collections()
        except Exception as exc:
            logger.exception("Failed to list collections")
            return _error(str(exc), status=500)
        return JsonResponse({"collections": names})

    if request.method == "POST":
        data = _parse_json_body(request)
        name = data.get("name", CHROMA_DEFAULT_COLLECTION)
        metadata = data.get("metadata", None)
        try:
            collection = get_or_create_collection(name=name, metadata=metadata)
        except Exception as exc:
            logger.exception("Failed to create collection '%s'", name)
            return _error(str(exc), status=500)
        return JsonResponse({
            "name": collection.name,
            "count": collection.count(),
        }, status=201)

    return _error("Method not allowed", status=405)


@csrf_exempt
def chroma_collection_detail(request: HttpRequest, name: str) -> JsonResponse:
    """Manage a single ChromaDB collection.

    ``GET``
        Return collection info (name, count).

    ``DELETE``
        Delete the collection and all its documents.
    """
    if request.method == "GET":
        try:
            collection = get_or_create_collection(name=name)
        except Exception as exc:
            logger.exception("Failed to get collection '%s'", name)
            return _error(str(exc), status=500)
        return JsonResponse({
            "name": collection.name,
            "count": collection.count(),
        })

    if request.method == "DELETE":
        try:
            delete_collection(name)
        except Exception as exc:
            logger.exception("Failed to delete collection '%s'", name)
            return _error(str(exc), status=500)
        return JsonResponse({"deleted": name})

    return _error("Method not allowed", status=405)


# ======================================================================
# ChromaDB — Documents
# ======================================================================

@csrf_exempt
def chroma_documents(request: HttpRequest, collection_name: str) -> JsonResponse:
    """Manage documents inside a ChromaDB collection.

    ``GET``
        Retrieve documents.  Query params:

        - ``ids`` (comma-separated)
        - ``where`` (JSON string)
        - ``limit`` (int)
        - ``offset`` (int)

    ``POST``
        Add one or more documents.

        Body (JSON)::

            {"documents": ["doc text 1", "doc text 2"],
             "metadatas": [{"source": "a"}, {"source": "b"}],  // optional
             "ids": ["id1", "id2"]}                            // optional (auto-generated)

    ``PUT``
        Update documents by id.

        Body (JSON)::

            {"ids": ["id1", "id2"],
             "documents": ["new text 1", "new text 2"],   // optional
             "metadatas": [{"source": "x"}, ...]}         // optional

    ``DELETE``
        Delete documents by id.

        Body (JSON)::

            {"ids": ["id1", "id2"]}
    """
    try:
        collection = get_or_create_collection(name=collection_name)
    except Exception as exc:
        logger.exception("Failed to get collection '%s'", collection_name)
        return _error(str(exc), status=500)

    # --- GET -----------------------------------------------------------
    if request.method == "GET":
        ids = None
        where = None
        limit = None
        offset = None

        if "ids" in request.GET:
            ids = request.GET["ids"].split(",")
        if "where" in request.GET:
            try:
                where = json.loads(request.GET["where"])
            except json.JSONDecodeError:
                return _error("Invalid JSON in 'where' param")
        if "limit" in request.GET:
            try:
                limit = int(request.GET["limit"])
            except ValueError:
                return _error("'limit' must be an integer")
        if "offset" in request.GET:
            try:
                offset = int(request.GET["offset"])
            except ValueError:
                return _error("'offset' must be an integer")

        try:
            result = get_documents(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                collection=collection,
            )
        except Exception as exc:
            logger.exception("Failed to get documents")
            return _error(str(exc), status=500)
        return JsonResponse(result, safe=False)

    # --- POST (add) ----------------------------------------------------
    if request.method == "POST":
        data = _parse_json_body(request)
        documents = data.get("documents")
        if not documents or not isinstance(documents, list):
            return _error("'documents' (list of strings) is required")

        metadatas = data.get("metadatas")
        ids = data.get("ids")

        try:
            add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                collection=collection,
            )
        except Exception as exc:
            logger.exception("Failed to add documents")
            return _error(str(exc), status=500)

        return JsonResponse({
            "added": len(documents),
            "collection": collection.name,
        }, status=201)

    # --- PUT (update) --------------------------------------------------
    if request.method == "PUT":
        data = _parse_json_body(request)
        ids = data.get("ids")
        if not ids or not isinstance(ids, list):
            return _error("'ids' (list of strings) is required")

        documents = data.get("documents")
        metadatas = data.get("metadatas")

        try:
            update_documents(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                collection=collection,
            )
        except Exception as exc:
            logger.exception("Failed to update documents")
            return _error(str(exc), status=500)

        return JsonResponse({"updated": len(ids)})

    # --- DELETE --------------------------------------------------------
    if request.method == "DELETE":
        data = _parse_json_body(request)
        ids = data.get("ids")
        if not ids or not isinstance(ids, list):
            return _error("'ids' (list of strings) is required")

        try:
            delete_documents(ids=ids, collection=collection)
        except Exception as exc:
            logger.exception("Failed to delete documents")
            return _error(str(exc), status=500)

        return JsonResponse({"deleted": len(ids)})

    return _error("Method not allowed", status=405)


# ======================================================================
# ChromaDB — Query
# ======================================================================

@csrf_exempt
def chroma_query(request: HttpRequest, collection_name: str) -> JsonResponse:
    """Query a ChromaDB collection by text.

    ``POST``

        Body (JSON)::

            {"query_texts": ["search query"],
             "n_results": 5,                                  // optional (default 5)
             "where": {"source": "wiki"},                     // optional metadata filter
             "where_document": {"$contains": "keyword"}}      // optional doc filter
    """
    if request.method != "POST":
        return _error("Method not allowed — use POST", status=405)

    data = _parse_json_body(request)
    query_texts = data.get("query_texts")
    if not query_texts or not isinstance(query_texts, list):
        return _error("'query_texts' (list of strings) is required")

    n_results = data.get("n_results", 5)
    where = data.get("where")
    where_document = data.get("where_document")

    try:
        collection = get_or_create_collection(name=collection_name)
        result = query_documents(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            collection=collection,
        )
    except Exception as exc:
        logger.exception("Failed to query collection '%s'", collection_name)
        return _error(str(exc), status=500)

    return JsonResponse(result)


# ======================================================================
# LLM — Generate
# ======================================================================

@csrf_exempt
def llm_generate(request: HttpRequest) -> JsonResponse | StreamingHttpResponse:
    """Generate text via the LLM (OpenRouter).

    ``POST``

        Body (JSON)::

            {"prompt": "What is the meaning of life?",
             "system": "You are a helpful assistant.",      // optional
             "model": "openai/gpt-4o",                      // optional
             "temperature": 0.7,                            // optional
             "max_tokens": 256,                             // optional
             "stream": false}                               // optional

        When ``stream`` is ``true`` the response is a Server-Sent Events
        stream with each token as a ``data`` event.
    """
    if request.method != "POST":
        return _error("Method not allowed — use POST", status=405)

    data = _parse_json_body(request)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return _error("'prompt' (non-empty string) is required")

    system = data.get("system")
    model = data.get("model")
    temperature = float(data.get("temperature", 0.7))
    max_tokens = data.get("max_tokens")
    stream = bool(data.get("stream", False))

    # Validate max_tokens type
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            return _error("'max_tokens' must be an integer")

    # --- Streaming path -------------------------------------------------
    if stream:
        try:
            token_stream = generate(
                prompt=prompt,
                system=system,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:
            logger.exception("Failed to start streaming generation")
            return _error(str(exc), status=500)

        def _sse_generator():
            """Yield SSE-format chunks from the LangChain token stream."""
            try:
                for chunk in token_stream:
                    # chunk is an AIMessageChunk; its .content is the delta text
                    delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if delta:
                        yield f"data: {json.dumps({'token': delta})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("Error during streaming")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        response = StreamingHttpResponse(
            _sse_generator(),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response

    # --- Non-streaming path ---------------------------------------------
    try:
        result = generate(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as exc:
        logger.exception("Failed to generate text")
        return _error(str(exc), status=500)

    return JsonResponse({
        "generated": result,
        "model": model or "openai/gpt-4o",
    })
