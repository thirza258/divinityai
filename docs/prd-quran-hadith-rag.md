# Product Requirements Document
# Islamic Grounded RAG — Quran & Hadith QA System

**Version:** 1.0.0
**Status:** Draft
**Author:** Thirzq
**Created:** 2026-07-04
**Last Updated:** 2026-07-04

---

## Table of Contents

1. [Overview](#1-overview)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [System Architecture](#3-system-architecture)
4. [Corpus & Knowledge Base](#4-corpus--knowledge-base)
5. [Pipeline Specification](#5-pipeline-specification)
   - 5.1 [Intent Router](#51-intent-router)
   - 5.2 [Scope Guard](#52-scope-guard)
   - 5.3 [Query Rewriting (HyDE + Sub-query)](#53-query-rewriting-hyde--sub-query)
   - 5.4 [Hybrid Retrieval](#54-hybrid-retrieval)
   - 5.5 [Reciprocal Rank Fusion](#55-reciprocal-rank-fusion)
   - 5.6 [Citation Verifier](#56-citation-verifier)
   - 5.7 [Evidence Sufficiency Check](#57-evidence-sufficiency-check)
   - 5.8 [Grounded Generation](#58-grounded-generation)
   - 5.9 [Safety Layer](#59-safety-layer)
6. [LLM Infrastructure (No-GPU)](#6-llm-infrastructure-no-gpu)
7. [Tech Stack](#7-tech-stack)
8. [Data Models](#8-data-models)
9. [API Design](#9-api-design)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Hallucination Mitigation Strategy](#11-hallucination-mitigation-strategy)
12. [Safety & Compliance Requirements](#12-safety--compliance-requirements)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Milestones & Phasing](#14-milestones--phasing)
15. [Research References](#15-research-references)
16. [Open Questions](#16-open-questions)

---

## 1. Overview

### 1.1 Problem Statement

General-purpose LLMs hallucinate Quranic verses and Hadith with alarming frequency. They fabricate surah numbers, misattribute narrators, conflate similar-sounding narrations, and generate plausible-but-false religious rulings. This is both intellectually dishonest and potentially harmful to Muslim users seeking guidance.

### 1.2 Solution

A purpose-built Retrieval-Augmented Generation (RAG) system that:
- Grounds **every answer exclusively** in a locked canonical corpus of Al-Quran and Hadith collections
- Attaches verifiable citations (`[Q 2:255]`, `[C Bukhari 52]`) to every claim
- Rejects or escalates queries that cannot be grounded in retrieved sources
- Runs entirely on free-tier LLM APIs (no GPU required)

### 1.3 Target Users

- Muslim individuals seeking Quranic or Hadith-based answers
- Islamic education platforms
- Developers building Islamic knowledge products
- Researchers in Islamic digital humanities

### 1.4 Core Design Principle

> **"If it's not in the corpus, don't say it."**
> Every factual claim in a generated answer must trace back to a specific retrieved chunk with a verifiable source tag.

---

## 2. Goals & Non-Goals

### Goals ✅

- Minimum hallucination for Quran and Hadith content
- Every answer cites exact source: surah+ayah or book+hadith number
- Supports Arabic, English, and Malay query input
- Intent-aware routing (verse lookup, hadith QA, fiqh guidance, calculation)
- Runs on CPU + free LLM APIs (no GPU dependency)
- Graceful degradation: "I don't have a grounded source" rather than fabricating
- Fatwa-boundary escalation for sensitive rulings

### Non-Goals ❌

- Not a fatwa-issuing system (it cites sources; it does not rule)
- Not a replacement for qualified Islamic scholars
- Not a general-purpose chatbot
- No support for tafsir commentary beyond retrieved texts (phase 1)
- No user account management (phase 1)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Gateway                         │
│           (FastAPI / Django REST)                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Pre-Retrieval Layer │
         │  Intent Router        │
         │  Scope Guard          │
         │  Query Rewriting      │
         │  (HyDE + Sub-query)   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Retrieval Layer     │
         │  BM25 (rank_bm25)     │
         │  Dense Embed (BGE-M3) │
         │  RRF Fusion           │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Validation Layer     │
         │  Cross-encoder Rerank │
         │  Citation Verifier    │
         │  (exact→fuzzy→sem)    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Generation Layer     │
         │  Evidence Check       │
         │  Grounded LLM Gen     │
         │  (context-only prompt)│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Safety Layer         │
         │  Hallucination Detect │
         │  Fatwa Boundary       │
         │  Content Guard        │
         └───────────┬───────────┘
                     │
              Final Answer
         (Answer + [Q*] + [C*] refs)
```

---

## 4. Corpus & Knowledge Base

### 4.1 Quran Corpus (QPC)

| Field | Detail |
|---|---|
| Source | King Fahd Complex Uthmani Quran (primary) |
| Translations | Saheeh International (EN), Muhammad Taqi-ud-Din (EN), Bahasa Indonesia (ID) |
| Total chunks | 6,236 ayahs (each ayah = one chunk) |
| Chunk metadata | `surah_number`, `ayah_number`, `surah_name_ar`, `surah_name_en`, `juz`, `makkiyya/madaniyya`, `text_ar`, `text_en` |
| Special handling | Diacritic variants indexed separately; normalized form used for fuzzy matching |

### 4.2 Hadith Corpus

| Collection | Arabic Title | Hadith Count (approx.) |
|---|---|---|
| Sahih Bukhari | صحيح البخاري | 7,563 |
| Sahih Muslim | صحيح مسلم | 7,190 |
| Sunan Abu Dawud | سنن أبي داود | 5,274 |
| Jami' at-Tirmidhi | جامع الترمذي | 3,956 |
| Sunan an-Nasa'i | سنن النسائي | 5,762 |
| Sunan Ibn Majah | سنن ابن ماجه | 4,341 |
| **Total** | | ~34,086 |

Each Hadith chunk metadata: `book`, `collection`, `hadith_number`, `chapter`, `narrator`, `grade` (sahih/hasan/da'if), `text_ar`, `text_en`, `source_tag`.

### 4.3 Source Tagging Convention

```
Quran:  [Q 2:255]          → Surah Al-Baqarah, Ayah 255
Hadith: [C Bukhari/52]     → Sahih Bukhari, Hadith 52
        [C Muslim/2564]    → Sahih Muslim, Hadith 2564
        [C AbuDawud/1420]  → Sunan Abu Dawud, Hadith 1420
```

### 4.4 Vector Index

- **Embedding model:** `BAAI/bge-m3` via HuggingFace Inference API (supports Arabic + multilingual)
- **Vector DB:** ChromaDB (local, CPU-only, persistent)
- **Collections:** `quran_ayahs`, `hadith_all`
- **BM25 index:** `rank_bm25` library, serialized to disk per corpus

---

## 5. Pipeline Specification

### 5.1 Intent Router

**Purpose:** Classify the incoming query into one of five categories to direct it to the appropriate retrieval and generation path.

**LLM call:** 1 call to Gemini 2.5 Flash, returns structured JSON.

**Prompt template:**
```
You are an Islamic query classifier.
Classify the query into exactly one category. Return JSON only, no explanation.

Categories:
- "quran_verse"    → user wants a specific ayah or surah reference
- "hadith"         → user wants a hadith or prophetic narration
- "fiqh"           → user wants Islamic jurisprudence guidance
- "calculation"    → user wants zakat, mirath (inheritance), prayer time math
- "off_domain"     → query is unrelated to Islam or Quran/Hadith

Query: "{query}"

Return: {"type": "<category>", "confidence": 0.0-1.0}
```

**Routing logic:**

| Intent | Retrieval target | Notes |
|---|---|---|
| `quran_verse` | QPC only | BM25 exact match prioritized |
| `hadith` | Hadith corpus only | Chain/narrator filter applied |
| `fiqh` | Both QPC + Hadith | Evidence synthesis required |
| `calculation` | Deterministic engine | No LLM generation for result |
| `off_domain` | ❌ Rejected | Return scope guard message |

---

### 5.2 Scope Guard

**Purpose:** Prevent the pipeline from attempting to answer off-domain queries that have no grounding in the corpus.

**Trigger:** Intent = `off_domain` OR confidence < 0.6 on any intent.

**Response:**
```json
{
  "status": "out_of_scope",
  "message": "This system only answers questions grounded in the Quran and authenticated Hadith collections. Your question appears to be outside this scope.",
  "suggestion": "Please rephrase with a specific Islamic topic."
}
```

---

### 5.3 Query Rewriting (HyDE + Sub-query)

**Purpose:** Improve recall by generating richer query variants before retrieval.

**Step A — HyDE (Hypothetical Document Embedding):**
```
You are a Quran and Hadith scholar.
Given the question: "{query}"
Write a short hypothetical passage (2–3 sentences) that would appear in an 
authentic Hadith or Quranic tafsir that directly answers this question.
Respond in the same language as the query. Arabic terms are welcome.
```

**Step B — Sub-query decomposition (for `fiqh` intent only):**
```
Decompose this Islamic jurisprudence question into 2–3 atomic sub-questions, 
each answerable from Quran or Hadith independently.
Return as JSON array of strings.
Query: "{query}"
```

**Output:** Up to 4 query variants fed to the retrieval layer (original + HyDE hypothesis + sub-queries).

---

### 5.4 Hybrid Retrieval

**For each query variant:**

| Retriever | Config | Notes |
|---|---|---|
| BM25 | `k=10`, `rank_bm25.BM25Okapi` | Applied to Arabic normalized text |
| Dense | `k=10`, ChromaDB cosine similarity | BGE-M3 1024-dim embeddings |

**Arabic normalization for BM25:**
- Strip tashkeel (diacritics)
- Normalize alef variants (`أ إ آ ا` → `ا`)
- Strip tatweel

**Per-query output:** Up to 20 chunks per corpus per query variant.

---

### 5.5 Reciprocal Rank Fusion

**Formula:**
```
RRF(doc) = Σ 1 / (k + rank_i)
```
where `k = 60` (standard default).

**Steps:**
1. Merge BM25 ranks + dense ranks per query variant
2. Apply RRF across all query variants
3. Top-N = 10 chunks per corpus (Quran) + Top-N = 10 chunks (Hadith)
4. Each chunk retains its `source_tag` throughout

---

### 5.6 Citation Verifier

**Purpose:** Verify that every cited verse/hadith in a generated answer actually exists in the canonical corpus, at the exact reference claimed.

**This is a Python-only step — no LLM call needed.**

**Cascade:**

```python
def verify_citation(cited_text: str, source_tag: str) -> VerificationResult:
    # Step 1: Exact string match
    canonical = corpus.get(source_tag)
    if canonical and cited_text == canonical:
        return VerificationResult(status="exact", score=1.0)

    # Step 2: Diacritic-normalized match
    if normalize(cited_text) == normalize(canonical):
        return VerificationResult(status="normalized", score=0.95)

    # Step 3: Fuzzy match (Levenshtein via rapidfuzz)
    score = fuzz.ratio(cited_text, canonical)
    if score >= 85:
        return VerificationResult(status="fuzzy", score=score/100)

    # Step 4: Semantic similarity fallback (LLM call)
    llm_verdict = llm_semantic_check(cited_text, canonical)
    if llm_verdict.match:
        return VerificationResult(status="semantic", score=llm_verdict.confidence)

    # Failed all checks → flag as hallucinated
    return VerificationResult(status="hallucinated", score=0.0)
```

**Policy:** Any citation with `status == "hallucinated"` is removed from context before generation.

---

### 5.7 Evidence Sufficiency Check

**Purpose:** Determine if retrieved chunks are sufficient to answer the query. If not, trigger an iterative sub-query loop rather than generating with weak evidence.

**LLM call:** Groq Llama 3.3 70B (fast inference).

**Prompt:**
```
Given these retrieved passages from Quran and Hadith:
{retrieved_chunks}

Can these passages sufficiently answer the question: "{query}"?
Return JSON: {"sufficient": true/false, "missing_aspect": "..." or null}
```

**Loop limit:** Maximum 2 re-retrieval iterations to prevent runaway API usage.

---

### 5.8 Grounded Generation

**Purpose:** Generate the final answer, constrained strictly to retrieved context.

**LLM call:** Gemini 2.5 Flash (1M context window).

**System prompt:**
```
You are an Islamic knowledge assistant. Your sole purpose is to answer 
questions using ONLY the Quran and Hadith passages provided to you.

RULES:
1. Answer ONLY from the provided context. Do not add any external knowledge.
2. Cite every Quranic reference as [Q surah:ayah], e.g. [Q 2:255]
3. Cite every Hadith as [C collection/number], e.g. [C Bukhari/52]
4. If the answer is not found in the provided context, respond with:
   "I do not have a grounded source for this in the provided passages."
5. Do not issue fatwas or definitive rulings. Present what the sources say.
6. If the question involves sensitive jurisprudence, add:
   "For a definitive ruling, please consult a qualified scholar."
7. Respond in the same language as the user's question.

Context:
{retrieved_chunks_with_source_tags}
```

---

### 5.9 Safety Layer

Three checks run in parallel on the generated answer before returning to the user.

**A. Hallucination Detector**

LLM call (Groq Llama 3.3 70B):
```
Check every Quran verse reference and Hadith citation in this answer.
Compare each against the provided source passages.
Return JSON:
{
  "hallucinated": true/false,
  "flagged_spans": [{"text": "...", "reason": "..."}]
}

Answer: {answer}
Source passages: {retrieved_chunks}
```

**B. Fatwa Boundary**

Rule-based pattern match against a keyword list of sensitive topics (medical, political, financial Islamic law, divorce rulings, inheritance disputes). If matched:
- Append standard disclaimer
- Log for review

**C. Content Guard**

Block answer if it:
- Contains claims not attributable to any retrieved chunk
- References sources outside the loaded corpus
- Makes absolutist doctrinal claims without scholarly qualification

---

## 6. LLM Infrastructure (No-GPU)

Full pipeline runs on free-tier LLM APIs — no GPU required.

### 6.1 Provider Assignment

| Stage | Provider | Model | Free Tier |
|---|---|---|---|
| Intent routing | Google AI Studio | `gemini-2.5-flash` | 1,500 req/day |
| HyDE generation | Google AI Studio | `gemini-2.5-flash` | shared pool |
| Scope guard check | Google AI Studio | `gemini-2.5-flash` | shared pool |
| Evidence sufficiency | Groq | `llama-3.3-70b` | 30 RPM |
| Final generation | Google AI Studio | `gemini-2.5-flash` | 1M ctx |
| Hallucination check | Groq | `llama-3.3-70b` | 30 RPM |
| Citation verifier | ❌ Python only | `rapidfuzz` | — |
| Embedding | HuggingFace Inference API | `BAAI/bge-m3` | Free (rate-limited) |

### 6.2 API Call Budget Per Query

| Step | LLM calls | Provider |
|---|---|---|
| Intent routing | 1 | Gemini Flash |
| HyDE | 1 | Gemini Flash |
| Evidence sufficiency | 1 | Groq |
| Final generation | 1 | Gemini Flash |
| Hallucination check | 1 | Groq |
| Re-retrieval (if needed) | 0–2 | Gemini Flash |
| **Max total** | **7** | — |

**Estimated daily capacity:** ~200–250 full queries/day at $0 cost.

### 6.3 Fallback Chain

```
Gemini 2.5 Flash  →  (rate limited)  →  OpenRouter DeepSeek R1 (free)
Groq Llama 3.3    →  (rate limited)  →  Cerebras Llama 3.3 70B (1M tokens/day)
```

---

## 7. Tech Stack

### Backend

| Component | Technology |
|---|---|
| Web framework | FastAPI (Python) |
| Task queue | Celery + Redis (for async pipeline) |
| Vector database | ChromaDB (persistent, CPU) |
| BM25 | `rank_bm25` |
| Fuzzy matching | `rapidfuzz` |
| Embedding inference | HuggingFace `InferenceClient` |
| LLM clients | `google-generativeai`, `groq`, `openai` (OpenRouter) |
| LLM orchestration | LlamaIndex `QueryPipeline` |
| Arabic NLP | `camel-tools` (normalization, tokenization) |
| Containerization | Docker + Docker Compose |

### Frontend (optional phase 2)

| Component | Technology |
|---|---|
| UI framework | React + TypeScript |
| State management | Zustand |
| Chat components | Custom (no external chat lib) |

### Infrastructure

| Component | Technology |
|---|---|
| Reverse proxy | Nginx |
| Environment config | `.env` + `pydantic-settings` |
| Logging | structlog + JSON output |
| Monitoring | Prometheus metrics endpoint |

---

## 8. Data Models

### 8.1 Query Request

```python
class QueryRequest(BaseModel):
    query: str                          # User's question
    language: Literal["ar", "en", "id"] = "en"
    max_sources: int = 5                # Max citations to return
    include_arabic: bool = True         # Include Arabic text in response
```

### 8.2 Retrieved Chunk

```python
class RetrievedChunk(BaseModel):
    source_tag: str                     # e.g. "Q:2:255" or "C:Bukhari:52"
    corpus: Literal["quran", "hadith"]
    text_ar: str
    text_en: str
    metadata: dict                      # surah/hadith-specific fields
    retrieval_score: float              # RRF score
    verification_status: Literal[
        "exact", "normalized", "fuzzy", "semantic", "hallucinated"
    ]
```

### 8.3 Query Response

```python
class QueryResponse(BaseModel):
    query: str
    intent: str                         # Detected intent
    answer: str                         # Generated answer with inline citations
    sources: List[RetrievedChunk]       # Verified sources used
    citations: List[str]                # ["Q 2:255", "C Bukhari/52"]
    safety: SafetyResult
    pipeline_meta: PipelineMeta         # Timing, LLM calls, iteration count
```

### 8.4 Safety Result

```python
class SafetyResult(BaseModel):
    hallucination_detected: bool
    flagged_spans: List[str]
    fatwa_boundary_triggered: bool
    disclaimer: Optional[str]
```

---

## 9. API Design

### Endpoints

```
POST   /api/v1/query          Query the Islamic RAG pipeline
GET    /api/v1/health         Health check
GET    /api/v1/corpus/stats   Corpus statistics (chunk count, collections)
POST   /api/v1/verify         Verify a specific citation against corpus
GET    /api/v1/metrics        Prometheus metrics
```

### POST /api/v1/query

**Request:**
```json
{
  "query": "What does the Quran say about patience?",
  "language": "en",
  "max_sources": 5
}
```

**Response:**
```json
{
  "query": "What does the Quran say about patience?",
  "intent": "quran_verse",
  "answer": "The Quran emphasizes patience (sabr) extensively. Allah says 'O you who have believed, seek help through patience and prayer. Indeed, Allah is with the patient.' [Q 2:153]. This is reinforced in [Q 39:10] where those who are patient will receive their reward without account.",
  "sources": [
    {
      "source_tag": "Q:2:153",
      "corpus": "quran",
      "text_ar": "يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ",
      "text_en": "O you who have believed, seek help through patience and prayer...",
      "verification_status": "exact",
      "retrieval_score": 0.94
    }
  ],
  "citations": ["Q 2:153", "Q 39:10"],
  "safety": {
    "hallucination_detected": false,
    "flagged_spans": [],
    "fatwa_boundary_triggered": false,
    "disclaimer": null
  }
}
```

---

## 10. Non-Functional Requirements

| Requirement | Target |
|---|---|
| End-to-end latency (p95) | < 8 seconds |
| Citation accuracy | ≥ 95% of citations pass exact/normalized verification |
| Hallucination rate | < 5% of answers contain a flagged span |
| Scope rejection accuracy | ≥ 98% of off-domain queries correctly rejected |
| Uptime | 99% (single instance, free-tier constraints) |
| Arabic query support | Full support (normalization + diacritic handling) |
| Daily query capacity | ≥ 200 full queries at $0 cost |
| Corpus coverage | All 6,236 Quran ayahs + 6 major Hadith collections |

---

## 11. Hallucination Mitigation Strategy

This system uses a **defense-in-depth** approach with four independent layers:

```
Layer 1:  Corpus lock       — LLM can ONLY see retrieved chunks, never training knowledge
Layer 2:  Source tagging    — Every chunk carries a verifiable source tag through the pipeline
Layer 3:  Citation verifier — Python exactness check before generation (no LLM bias)
Layer 4:  Post-gen check    — LLM verifies its own output against the source passages
```

**Key insight:** The citation verifier (Layer 3) is the most reliable guard because it uses deterministic Python string matching, not another LLM call. A hallucination by the generation LLM will be caught by the post-gen check (Layer 4) even if it passes Layer 3 (e.g., fabricated attribution).

---

## 12. Safety & Compliance Requirements

### 12.1 Religious Sensitivity

- The system **never issues fatwas** — it presents what sources say, not rulings
- Sensitive jurisprudence topics (divorce, inheritance, usury, medical) always trigger the fatwa boundary disclaimer
- Contradictory scholarly positions are presented as-is without resolution

### 12.2 Source Integrity

- Corpus is locked at build time; no runtime modification
- Version-controlled corpus with checksums
- Any source outside the 6 canonical Hadith collections is out of scope for phase 1

### 12.3 Content Moderation

- Queries attempting to extract rulings supporting violence, harm, or extremism are rejected at scope guard
- Rejection is logged for review

### 12.4 Transparency

- Every response includes which sources were used
- Users can request the raw retrieved chunks via `include_sources: true`

---

## 13. Evaluation Metrics

### 13.1 Retrieval Quality

| Metric | Formula | Target |
|---|---|---|
| Recall@10 | Relevant chunks in top 10 | ≥ 0.85 |
| MRR | Mean Reciprocal Rank | ≥ 0.75 |
| Citation Precision | Verified citations / Total citations | ≥ 0.95 |

### 13.2 Generation Quality

| Metric | Tool | Target |
|---|---|---|
| Faithfulness | RAGAS `faithfulness` | ≥ 0.90 |
| Answer Relevancy | RAGAS `answer_relevancy` | ≥ 0.85 |
| BERTScore F1 | `bert-score` (Arabic model) | ≥ 0.80 |
| Hallucination Rate | Span detection F1 | ≤ 0.05 |

### 13.3 Safety Metrics

| Metric | Target |
|---|---|
| Off-domain rejection rate | ≥ 0.98 |
| Fatwa boundary trigger accuracy | ≥ 0.90 |
| False positive (safe answer blocked) | ≤ 0.02 |

### 13.4 Evaluation Dataset

A curated benchmark of 500 Islamic QA pairs is required:
- 200 Quran verse lookup questions
- 150 Hadith retrieval questions
- 100 fiqh guidance questions (with expert-annotated ground truth)
- 50 off-domain questions (should be rejected)

---

## 14. Milestones & Phasing

### Phase 1 — Core Pipeline (Weeks 1–3)
- [ ] Corpus ingestion: Quran + 2 Hadith collections (Bukhari, Muslim)
- [ ] ChromaDB + BM25 index built
- [ ] Hybrid retrieval + RRF implemented
- [ ] Basic generation with context-only prompt
- [ ] Citation verifier (exact + fuzzy)
- [ ] FastAPI `/query` endpoint working

### Phase 2 — Full Pipeline (Weeks 4–6)
- [ ] All 6 Hadith collections ingested
- [ ] Intent router + scope guard deployed
- [ ] HyDE + sub-query decomposition
- [ ] Post-generation hallucination detector
- [ ] Fatwa boundary trigger
- [ ] Multi-language support (AR/EN/ID)

### Phase 3 — Robustness (Weeks 7–8)
- [ ] Evaluation benchmark (500 QA pairs)
- [ ] RAGAS evaluation pipeline
- [ ] Provider fallback chain
- [ ] Rate limit management
- [ ] Docker Compose full stack

### Phase 4 — Frontend (Weeks 9–10)
- [ ] React/TypeScript chat interface
- [ ] Citation rendering with source popover
- [ ] Arabic text rendering (RTL support)
- [ ] Mobile responsive

---

## 15. Research References

| Paper | Contribution to this system | arXiv / Source |
|---|---|---|
| RAG Survey (Gao et al.) | Pipeline taxonomy | `arXiv:2312.10997` |
| HyDE (Gao et al.) | Hypothetical document generation | `arXiv:2212.10496` |
| RAG-Fusion | RRF multi-query fusion | `arXiv:2402.03367` |
| Adaptive-RAG | Intent-based routing | `arXiv:2403.14403` |
| FLARE | Iterative re-retrieval loop | `arXiv:2305.06983` |
| Self-RAG | Reflection tokens, on-demand retrieval | `arXiv:2310.11511` |
| CRAG | Corrective retrieval + evidence sufficiency | `arXiv:2401.15884` |
| Fanar-Sadiq | Intent routing for Islamic QA | `arXiv:2603.08501` |
| FARSIQA / FAIR-RAG | Iterative evidence loop, 97% negative rejection | `arXiv:2510.25621` |
| Canal Mesh | Sidecar-less service mesh (infra ref) | SIGCOMM 2024 |
| Microservice Granularity | Service Weaver framework | `arXiv:2404.09357` |

---

## 16. Open Questions

| # | Question | Priority | Owner |
|---|---|---|---|
| 1 | Which Arabic tokenizer is most accurate for classical Quranic Arabic: `camel-tools` vs `farasa` vs `mishkal`? | High | — |
| 2 | Should Tafsir (Quranic commentary) be included in Phase 1 as a third corpus, or Phase 2? | Medium | — |
| 3 | How to handle multi-madhab (school of law) questions where Hanafi, Maliki, Shafi'i, Hanbali opinions differ? | High | — |
| 4 | Should BGE-M3 be fine-tuned on an Islamic QA dataset, or is zero-shot sufficient for recall targets? | Medium | — |
| 5 | How to handle hadith grading (da'if narrations)? Should weak hadith be excluded from corpus or labeled? | High | — |
| 6 | Rate limit management strategy: single Redis counter or per-provider queues? | Low | — |
| 7 | What is the minimum viable evaluation dataset size for reliable RAGAS scores? | Medium | — |

---

*This PRD is a living document. Update with implementation decisions as the project progresses.*
