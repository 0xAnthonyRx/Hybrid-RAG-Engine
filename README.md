# Enterprise Hybrid RAG Engine

A two-stage retrieval and question-answering service built on **PostgreSQL (`pgvector`)**, **Reciprocal Rank Fusion (RRF)**, and **local Cross-Encoder Reranking (`FlashRank`)**.

Designed to eliminate retrieval failure modes on exact alphanumeric identifiers (SKUs, invoice IDs, serial codes) while maintaining semantic recall on complex, paraphrased domain queries.

## Table of Contents

- [The Problem](#the-problem-why-naive-vector-search-fails-in-production)
- [Architecture & Mathematical Foundations](#architecture--mathematical-foundations)
- [Retrieval Benchmark](#retrieval-benchmark-empirical-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Production Verification](#production-verification-sample-output)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## The Problem: Why Naive Vector Search Fails in Production

Standard RAG architectures rely entirely on dense vector similarity (bi-encoders). In enterprise environments, this creates three major failure modes:

1. **Alphanumeric Loss (Low Precision on Exact Tokens):** Vector embeddings compress text into dense semantic coordinates. Searching for `"INV-2026-904"` or part serial `"PUMP-TX8-LUB"` yields geometric approximations of concepts like *"logistics"* or *"machinery"*, often returning irrelevant invoices while missing the target file.

2. **Vocabulary Mismatch (Low Recall on Keywords):** Traditional keyword search (BM25 / Full-Text Search) fails on natural language paraphrases. Searching for *"how to prevent birds from dying in high temperatures"* returns zero hits if the standard operating procedure is titled *"Broiler thermal management and heat stress mitigation"*.

3. **Context Pollution ("Lost in the Middle"):** Feeding 10–20 raw vector candidate chunks directly into an LLM degrades reasoning, triggers hallucinations, and inflates token costs by up to 80%.

This engine solves these issues using a **Two-Stage Hybrid Architecture**:

```mermaid
flowchart TD
    Q[Incoming User Query] --> V[Dense Vector Search<br/>HNSW, 768d Gemini Embeddings<br/>Cosine Distance]
    Q --> K[Sparse Keyword Search<br/>GIN, English tsvector<br/>Sanitized Boolean Query]
    V --> RRF[Stage 1: SQL-Level RRF Fusion<br/>Single CTE Full Outer Join]
    K --> RRF
    RRF --> RR[Stage 2: Cross-Encoder Rerank<br/>FlashRank Local CPU ONNX<br/>Threshold Cutoff ≥ 0.2]
    RR --> G[Grounded Answer Generation<br/>Gemini 3.8 Flash via LiteLLM<br/>Strict Inline Source Citations]
```

## Architecture & Mathematical Foundations

### 1. Dual Indexing in PostgreSQL

The storage layer runs inside PostgreSQL with zero third-party vector database dependencies:

- **Dense Vectors:** Stored in a `vector(768)` column indexed via **HNSW (Hierarchical Navigable Small World)** graphs with `vector_cosine_ops`. Provides sub-10ms approximate nearest neighbor queries without requiring index retraining.
- **Sparse Lexical Tokens:** Stored in an auto-generated, stored `tsvector` column indexed with a **GIN (Generalized Inverted Index)**.
- **Metadata Routing:** Uses native PostgreSQL `JSONB` for zero-overhead structural pre-filtering.

### 2. Reciprocal Rank Fusion (RRF)

Combining raw cosine distance (0.0 ≤ dist ≤ 2.0) with PostgreSQL `ts_rank` (0.0 ≤ rank < ∞) by simple addition is mathematically invalid due to incompatible scales and distributions.

Instead, candidate documents from each retrieval pipeline are ranked independently, and their relative positions are blended using RRF:

$$
RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}
$$

Where:

- M = {Vector Search, Keyword Search}
- r_m(d) is the document's 1-based rank position in method m
- k = 60 (a common, tunable smoothing constant preventing high-rank dominance)

The entire fusion executes directly inside the database engine via a single SQL query leveraging Common Table Expressions (CTEs), window functions (`ROW_NUMBER()`), and a `FULL OUTER JOIN`.

### 3. Stage 2 Local Cross-Encoder Reranking

While bi-encoders compress queries and documents into separate vectors, cross-encoders feed the query and candidate passage simultaneously through cross-attention layers:

$$
\text{Score} = \text{CrossEncoder}(\text{Query}, \text{Passage})
$$

- **Inference Engine:** `FlashRank` running a quantized `ms-marco-TinyBERT-L-2-v2` model locally on CPU via ONNX Runtime (~5ms execution latency, zero API cost).
- **Confidence Floor Filtering:** Implements a strict cutoff threshold (`min_score = 0.20`). Candidates scoring below this threshold (e.g., zero-relevance artifacts from broad retrieval) are dropped before reaching the LLM context window.

## Retrieval Benchmark: Empirical Results

The pipeline was validated against two adversarial query profiles:

### Benchmark Data Matrix

| Query Type | Test Query | Pure Vector Search | Pure Keyword Search (GIN) | Hybrid RRF (Stage 1) | Two-Stage (RRF + FlashRank) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact SKU / ID** | `"Find invoice INV-2026-904"` | Distance: `0.3007` (Rank 1) | Score: `0.0304` (Rank 1) | **RRF: `0.03279` (Rank 1)** *(Both ranks reinforce)* | **Final Score: `0.8685`** *(Irrelevant passages dropped to 0.0)* |
| **Conceptual Paraphrase** | `"How do we keep the birds from dying when it is too hot?"` | Distance: `0.2800` (Rank 1) | **0 hits** *(No literal token overlap)* | **RRF: `0.01639` (Rank 1)** *(Degrades gracefully to vector rank)* | **Final Score: `0.8410`** *(High-confidence thermal SOP match)* |

### Key Findings

1. **The Reinforcement Effect:** On alphanumeric lookups where both search models agree, RRF doubles the score (1/61 + 1/61 ≈ 0.03279), ensuring exact records dominate the candidate pool.
2. **Graceful Degradation:** When natural language queries contain zero lexical overlap with the stored SOP, the keyword query safely yields 0.0, allowing the semantic vector branch to deliver the correct document at Rank 1.
3. **Context Compression:** FlashRank consistently scores non-relevant Stage 1 candidates at `0.0`, stripping them before LLM injection and preventing context pollution.

## Project Structure

```
├── schema.sql           # PostgreSQL DDL: pgvector, HNSW, GIN, and tsvector definitions
├── init_db.py           # Database migration and pgvector extension verification script
├── ingest.py            # Batch document ingestion & 768d vector generation
├── search.py            # Vector, Keyword, and Hybrid RRF SQL retrieval functions
├── benchmark.py         # Comparative benchmark harness (Vector vs Keyword vs Hybrid)
├── rag.py               # Two-stage retrieval pipeline, FlashRank reranking & grounded generation
├── requirements.txt     # Python runtime dependencies
├── .env.example         # Template for environment configuration
└── README.md            # Architecture, math, benchmarks, and deployment docs
```

## Getting Started

### 1. Prerequisites

- Python 3.11+
- Docker Engine
- Google Gemini API Key (or OpenAI / local model equivalent)

### 2. Environment Setup

Clone the repository and initialize the virtual environment:

```bash
git clone https://github.com/<your-username>/hybrid-rag-engine.git
cd hybrid-rag-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` based on `.env.example`:

```bash
cp .env.example .env
```

Populate `.env`:

```ini
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vectordb"
GEMINI_API_KEY="your-api-key-here"
EMBEDDING_MODEL="gemini/gemini-embedding-001"
EMBEDDING_DIM=768
GENERATION_MODEL="gemini/gemini-3.8-flash"
```

### 3. Launch Persistent PostgreSQL with pgvector

Start a persistent PostgreSQL 17 container configured with a named Docker volume:

```bash
docker run -d \
  --name pgvector-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectordb \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

Apply the database schema and indexes:

```bash
python3 init_db.py
```

### 4. Ingest Sample Documents

Embed and index the operational corpus:

```bash
python3 ingest.py
```

### 5. Run Benchmarks & Full Pipeline

Run the multi-retriever benchmark:

```bash
python3 benchmark.py
```

Execute the full two-stage RAG generation pipeline:

```bash
python3 rag.py
```

## Production Verification: Sample Output

```text
Processing Query: 'What is the specific maintenance schedule and lubricant for our water pump?'
-------------------------------------------------------
Stage 1 (Hybrid RRF): Retrieved 5 candidates from Postgres.
Stage 2 (FlashRank): Reranked down to top 1 most relevant passages:
- [MAN-PUMP-TX8] Cross-Encoder Score: 0.8685

--- Synthesized Answer with Grounded Citations ---
Based on Document [MAN-PUMP-TX8], the specific maintenance schedule and lubricant requirements for the automated irrigation pump are:

* Maintenance Schedule: Requires an oil change every 250 operating hours [MAN-PUMP-TX8].
* Lubricant: Use only 15W-40 industrial lubricant (Part serial: PUMP-TX8-LUB) [MAN-PUMP-TX8].

Sources Cited: ['MAN-PUMP-TX8']
```

## Limitations & Future Work

- **Small Corpus:** Currently validated on 5 documents. Scaling to 10k+ may require tuning candidate limits and batch reranking.
- **Single-Language:** English-only stemming and embeddings. Multi-language support would require model swaps.
- **LLM-based Generation:** Costs scale with query volume; local model substitution is possible.


MIT
