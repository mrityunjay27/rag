# RAG Project - Evaluation Progress Report

## Overview

This document tracks the progress of the RAG (Retrieval-Augmented Generation) project, focusing on the evaluation framework built so far. It covers the architecture, concepts implemented, and how different evaluation metrics work together.

## Project Architecture So Far

The RAG system follows a standard pipeline:

```text
RAG SYSTEM
    │
    ├──────────────┬─────────────────┐
    │              │                 │
    ↓              ↓                 ↓
  RETRIEVAL    EVALUATION        GENERATION
    │              │                 │
    ↓              ↓                 ↓
Search (Vector)  Metrics           LLM Answer
```

### 1. Retrieval Layer (`app/retrieval/`)
- **`search.py`**: Implements vector similarity search using pgvector
  - Uses cosine distance (`<=>` operator in pgvector)
  - Converts similarity: `1 - cosine_distance` to get similarity score
  - Supports filtering by source, similarity threshold, and top_k
  - Returns tuples: `(content, source, chunk_index, similarity)`

### 2. Embeddings Layer (`app/embeddings/`)
- **`embedder.py`**: Generates embeddings using SentenceTransformer
  - Model: `all-MiniLM-L6-v2` (lightweight, efficient)

### 3. Database Layer (`app/db/`)
- **`database.py`**: PostgreSQL connection with pgvector extension
- **`schema.sql`**: Database schema definition
- **`document_chunks` table**: Stores chunked documents with embeddings (vector type)

### 4. Data Ingestion (`app/ingestion/`)
- **`chunker.py`**: Document chunking logic
- **`loader.py`**: Document loading utilities
- **`embed_and_store.py`**: Embeds chunks and stores in database

### 5. Generation Layer (`app/generation/`)
- **`generator.py`**: LLM-based answer generation using retrieved context (Groq)
  - Model: `openai/gpt-oss-120b`
  - Prompts LLM with query + context to produce grounded answers

### 6. Evaluation Layer (`app/evaluation/`)
The evaluation module is organized into modular metric files. Each retrieval metric is isolated in its own file, and LLM-judge based metrics are also modular.

## Evaluation Concepts & Implementation

### RETRIEVAL EVALUATION

These metrics assess whether we retrieved the right context from the vector store.

#### 1. Recall@K (`recall.py`)
**What it means:** Did we retrieve the relevant evidence within the top K results?

**Question:** "Did we find the needed evidence?"

**How it works:** Binary (1 or 0) - returns 1 if the expected evidence appears in any of the top K retrieved chunks (case-insensitive substring match).

**Formula:** `Recall@K = 1 if relevant chunk is in top K, else 0`

**Example:**
- Relevant chunk is at rank 3
- K=3: Recall@3 = 1 (found)
- K=2: Recall@2 = 0 (not found)

**Goal:** Don't miss important information.

#### 2. Precision@K (`precision.py`)
**What it means:** Of the K chunks we retrieved, how many were actually relevant?

**Question:** "How much of what we retrieved is relevant?"

**How it works:** Fraction of retrieved chunks that contain the expected evidence (substring match).

**Formula:** `Precision@K = (number of relevant chunks in top K) / K`

**Example:**
- Top 3 chunks: [relevant, irrelevant, irrelevant] → Precision@3 = 1/3 = 33.3%

**Goal:** Don't send too much irrelevant context to the LLM.

#### 3. Mean Reciprocal Rank (MRR) - Reciprocal Rank (`reciprocal_rank.py`)
**What it means:** How high did the first relevant result appear in the ranking?

**Question:** "How high did relevant evidence rank?"

**How it works:** Returns 1/rank of the first relevant chunk found (1-based indexing).

**Formula:** `RR = 1 / (position of first relevant chunk)`, MRR is averaged across queries.

**Example:**
- First relevant at rank 1 → RR = 1.0
- First relevant at rank 2 → RR = 0.5
- First relevant at rank 3 → RR = 0.33
- Not found → RR = 0.0

**Goal:** Get relevant information as high as possible in the ranking.

### CONTEXT EVALUATION

#### 4. Context Relevance (`context_relevance.py` + `context_relevance_metric.py`)
**What it means:** Is the retrieved context actually useful for answering the question (beyond just containing matching text)?

**Question:** "Is the context useful for the query?"

**How it works:** LLM-as-a-judge using Groq API.
- **Judge model:** `openai/gpt-oss-120b` (via Groq)
- **Input:** Query + Retrieved Chunk
- **Output:** Binary YES/NO (case-normalized)
- **Temperature:** 0 for deterministic evaluation

**Why this matters:** Embedding similarity can be high but the chunk might not be semantically useful for answering the specific question.

**Implementation:**
- `context_relevance.py`: Core LLM judge function (`evaluate_context_relevance(query, context)`)
- `context_relevance_metric.py`: Applies judge to all retrieved chunks and computes percentage of relevant chunks

**Formula:** `Context Relevance@K = (number of relevant chunks in top K) / K`

**Goal:** Evaluate the quality/usefulness of retrieved context.

### GENERATION EVALUATION

#### 5. Faithfulness (`faithfulness.py`)
**What it means:** Is the generated answer supported by the retrieved context?

**Question:** "Is the answer supported by the context?"

**How it works:** LLM-as-a-judge comparing context vs generated answer.
- **Judge model:** `openai/gpt-oss-120b` (via Groq)
- **Input:** Context + Generated Answer
- **Output:** Binary YES/NO
- Returns YES only if the answer contains no factual claims unsupported by the context

**Example:**
- Context: "Refresh tokens obtain a new access token."
- Answer: "Refresh tokens obtain a new access token." → Faithful (YES)
- Answer: "Refresh tokens expire after 30 days." → Not faithful (NO - not stated in context)

**Goal:** Detect unsupported claims/hallucinations.

#### 6. Answer Correctness (`answer_correctness.py`)
**What it means:** Is the final answer actually correct from the user's perspective?

**Question:** "Is the answer actually correct?"

**How it works:** LLM-as-a-judge comparing Question + Expected Answer + Generated Answer.
- **Judge model:** `openai/gpt-oss-120b` (via Groq)
- **Input:** Question + Expected Answer + Generated Answer
- **Output:** Binary YES/NO
- Evaluates semantic correctness, not exact string matching

**Goal:** Evaluate the final answer's correctness.

## How It All Fits Together

```text
                         RAG Pipeline
                              │
                 ┌────────────┴────────────┐
                 │                         │
            RETRIEVAL                GENERATION
                 │                         │
          ┌──────┼──────┐             ┌────┴─────┐
          ↓      ↓      ↓             ↓          ↓
       Recall Precision MRR      Faithfulness Correctness
          │      │      │             │          │
          └──────┼──────┘             └──────────┘
                 ↓                         ↑
          Context Relevance               │
                 │                        │
                 └──── Retrieved Context ─┘
```

**Metric Categorization:**

| Metric | Evaluates | Core Question |
|---|---|---|
| Recall@K | Retrieval (completeness) | Did we find the needed evidence? |
| Precision@K | Retrieval (precision) | How much retrieved content is relevant? |
| MRR/RR | Ranking quality | How high did relevant evidence rank? |
| Context Relevance | Retrieved context utility | Is the context useful for the query? |
| Faithfulness | Generation groundedness | Is the answer supported by the context? |
| Answer Correctness | Final output quality | Is the answer actually correct? |

## File Structure

```text
app/evaluation/
├── __init__.py
├── __pycache__/
├── answer_correctness.py      # LLM judge for answer correctness
├── context_relevance.py       # LLM judge for chunk relevance
├── context_relevance_metric.py # Metric computation using LLM judge
├── dataset.py                # Evaluation dataset
├── faithfulness.py           # LLM judge for faithfulness
├── precision.py              # Precision@K calculation
├── recall.py                 # Recall@K calculation
├── reciprocal_rank.py        # Reciprocal Rank calculation
└── test_retrieval.py          # Main retrieval evaluation runner
```

## Running Evaluations

### Retrieval Evaluation
```bash
cd /Users/mrityunjay/Workspace/RAG\ Projects/rag
python app/evaluation/test_retrieval.py
```

This runs retrieval evaluation for k=[1, 2], computing:
- Recall@K (per query + overall)
- Precision@K (per query + overall)
- Reciprocal Rank (per query + MRR@K overall)
- Context Relevance@K (per query via LLM judge + overall percentage)

## Key Learnings

1. **Separation of Concerns**: Each metric is in its own module (recall, precision, reciprocal_rank) for better maintainability and reusability.

2. **Retrieval vs Generation**: 
- Retrieval-focused (Recall/Precision/MRR/Context Relevance) validate if we retrieve good context
- Generation-focused (Faithfulness/Correctness) validate if the LLM uses and answers correctly

3. **LLM-as-a-Judge**: Context Relevance, Faithfulness, and Answer Correctness leverage LLMs for nuanced semantic judgments beyond exact matching.

4. **Position Matters**: MRR emphasizes ranking - getting the right answer first is critical.

5. **Trade-offs**: Higher K improves Recall but typically reduces Precision; balance based on context window limits and noise tolerance.

6. **Grounding is Key**: Faithfulness ensures answers don't hallucinate beyond retrieved context.
