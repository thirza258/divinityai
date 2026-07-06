# Embedding Configuration Changes

## Overview

The embedding pipeline has been updated to use **Ollama** for embedding generation instead of the previous embedding provider.

### Embedding Provider

* **Provider:** Ollama
* **Embedding Model:** `embeddinggemma`

All embedding requests should be sent to the Ollama instance.

---

## Ollama Configuration

If the application runs on the host machine:

```text
http://localhost:11434
```

If the application runs inside Docker and Ollama runs on the host:

```text
http://host.docker.internal:11434
```

---

## ChromaDB Configuration

The ChromaDB server is hosted outside the application container.

### Host

If running directly on the host:

```text
http://localhost:8040
```

If the application runs inside Docker:

```text
http://host.docker.internal:8040
```

---

## Collections

Use the following ChromaDB collections:

| Dataset | Collection Name     |
| ------- | ------------------- |
| Quran   | `quran_collection`  |
| Hadith  | `hadith_collection` |

---

## Retrieval Flow

1. Receive the user query.
2. Generate query embeddings using **Ollama** with the `embeddinggemma` model.
3. Query ChromaDB using the generated embedding.
4. Retrieve documents from:

   * `quran_collection` for Quran documents.
   * `hadith_collection` for Hadith documents.
5. Pass the retrieved context to the LLM for response generation.

---

## Example Configuration

```yaml
embedding:
  provider: ollama
  base_url: http://host.docker.internal:11434
  model: embeddinggemma

chromadb:
  host: http://host.docker.internal:8040
  collections:
    quran: quran_collection
    hadith: hadith_collection
```

> **Note:** If the application is not running inside Docker, replace `host.docker.internal` with `localhost`.
