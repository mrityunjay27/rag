# RAG — Complete Learning Notes

Personal study notes covering everything learned about **Retrieval-Augmented Generation**, in the order it was learned.

**Scope:** Concepts and mental models only — implementation, code and library usage are intentionally excluded.
**Context:** Learner already knows Python, Django, PostgreSQL, Kafka, Redis and microservices, so RAG was studied at an engineering/deep level rather than as "how to call an LLM API".

---

## Table of Contents

1. [Orientation — Why and How RAG Should Be Learned](#1-orientation--why-and-how-rag-should-be-learned)
2. [Lesson 1 — LLM Fundamentals](#2-lesson-1--llm-fundamentals)
3. [Lesson 2 — Tokens and Tokenization](#3-lesson-2--tokens-and-tokenization)
4. [Lesson 3 — Transformers and Attention](#4-lesson-3--transformers-and-attention)
5. [Side Question — What is a Feed-Forward Network?](#5-side-question--what-is-a-feed-forward-network)
6. [Lesson 4 — Embeddings](#6-lesson-4--embeddings)
7. [Clarification — We Store Both the Chunk Text and the Embedding](#7-clarification--we-store-both-the-chunk-text-and-the-embedding)
8. [Lesson 5 — Vector Search and Vector Databases](#8-lesson-5--vector-search-and-vector-databases)
9. [Lesson 6 — The RAG Pipeline (Conceptually)](#9-lesson-6--the-rag-pipeline-conceptually)
10. [Lesson 7 — Chunking Strategies](#10-lesson-7--chunking-strategies)
11. [Lesson 8 — Embedding Models in Practice](#11-lesson-8--embedding-models-in-practice)
12. [Lesson 9 — Retrieval: Top-K, Filtering, Hybrid Search, Reranking](#12-lesson-9--retrieval-top-k-filtering-hybrid-search-reranking)
13. [Clarification — What Exactly Is a Reranker?](#13-clarification--what-exactly-is-a-reranker)
14. [Decision Point — Start Implementing](#14-decision-point--start-implementing)
15. [Stack Decisions — Embedding and Generation Providers](#15-stack-decisions--embedding-and-generation-providers)
16. [Roadmap Ahead — Not Yet Covered](#16-roadmap-ahead--not-yet-covered)
17. [Master Mental Models and Reference Tables](#17-master-mental-models-and-reference-tables)
18. [Common Misconceptions — Correct Answers](#18-common-misconceptions--correct-answers)
19. [Self-Check Question Bank](#19-self-check-question-bank)

---

## 1. Orientation — Why and How RAG Should Be Learned

### 1.1 RAG as a pipeline

RAG is best understood as a sequence of stages, not a single technique.

```text
          ┌──────────────────────┐
          │      Documents       │
          │ PDFs / DB / Web / API│
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │   Document Parsing   │
          │ clean / extract text │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │      Chunking        │
          │  split into chunks   │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │      Embeddings      │
          │ text → vectors       │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │     Vector DB        │
          │ pgvector / Qdrant    │
          └──────────┬───────────┘
                     │
               User Question
                     │
                     ▼
          ┌──────────────────────┐
          │       Retrieval      │
          │ semantic / keyword   │
          │ hybrid / reranking   │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │    Context Builder   │
          │ question + chunks    │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │         LLM          │
          └──────────┬───────────┘
                     ▼
               Final Answer
```

### 1.2 The core learning rule

**Do not** think of RAG as:

> "LLM + Vector DB"

That is the beginner definition.

A better mental model is:

> **RAG = information retrieval system + context construction + LLM generation**

### 1.3 The six interesting engineering problems in RAG

```text
Can I find the right information?
            ↓
Can I rank it correctly?
            ↓
Can I fit the useful information
into the context window?
            ↓
Can I prevent the LLM from
making things up?
            ↓
Can I measure whether it works?
            ↓
Can I operate it cheaply and reliably?
```

### 1.4 Learning rule adopted

- **Do not start with LangChain / LlamaIndex.** Understand what happens underneath first.
- The first RAG system should be hand-built, so frameworks are understood rather than memorised.
- Chunking needs to be understood *extremely well* — it is one of the areas where production RAG differs most from toy RAG.

### 1.5 The original 10-phase roadmap

| Phase | Topic | Key ideas to learn |
|---|---|---|
| **1** | LLM fundamentals | Tokens, context window, prompting, temperature, system/user messages, inference, hallucination, structured output, function/tool calling |
| **2** | Embeddings | Embedding models, dimensions, cosine similarity, Euclidean distance, dot product, normalization, semantic similarity |
| **3** | Vector databases | Vector indexes, HNSW, IVFFlat, approximate nearest-neighbor search, metadata filtering, dimensions, similarity thresholds |
| **4** | Document ingestion | Fixed-size / token-based / sentence / recursive / semantic chunking, overlap, parent-child chunks, document structure, metadata |
| **5** | Basic RAG | First complete RAG system, built by hand |
| **6** | Retrieval | Dense retrieval, sparse retrieval (BM25), hybrid retrieval |
| **7** | Reranking | Cross-encoder reranking, bi-encoder retrieval, recall vs precision, top-K, reranking |
| **8** | Advanced RAG | Query transformation, multi-query RAG, HyDE, parent-child retrieval, contextual retrieval, Graph RAG |
| **9** | RAG evaluation | Recall@K, Precision@K, MRR, NDCG, context relevance, context recall, answer relevance, faithfulness, groundedness |
| **10** | Production RAG | Redis caching, async ingestion, Kafka events, background workers, rate limiting, auth, multi-tenancy, observability, cost tracking, streaming, retries, dead-letter queues, document versioning |

*Progress so far: Phases 1 and 2 are complete, plus most of the Phase 4–6 material (chunking, retrieval, reranking). Phase 9 evaluation has not been taught yet.*

### 1.6 Why LLMs need RAG

A company typically holds knowledge the LLM was never trained on:

```text
Employee handbook
Product documentation
API documentation
Architecture documents
Internal FAQs
Support tickets
```

Asking an LLM *"What is our company's reimbursement policy?"* does not magically grant it access to internal documents. RAG retrieves those passages and places them into the context.

**The LLM is not the database. The retrieval system supplies the information.**

### 1.7 Recommended project direction (portfolio-grade)

Rather than a generic "chat with PDF", the suggested target is an **Enterprise Knowledge RAG** service: an API gateway in front of a RAG API, with a retrieval service and an LLM service, PostgreSQL + pgvector for storage and Redis for caching. Ingestion is driven by a Kafka `document.uploaded` event consumed by an ingestion worker (parse → chunk → embed → store).

Suggested version ladder:

| Version | Capability added |
|---|---|
| V1 | Basic vector RAG |
| V2 | Metadata filtering |
| V3 | Hybrid search |
| V4 | Reranking |
| V5 | Query rewriting |
| V6 | Evaluation |
| V7 | Streaming |
| V8 | Caching |
| V9 | Multi-tenant RAG |
| V10 | Production observability |

Production architecture considerations from Phase 10 that map onto existing backend experience: API gateway → RAG API service (retriever / LLM service / metadata) → vector DB, plus caching, async ingestion, Kafka events, background workers, rate limiting, authentication, multi-tenancy, observability, cost tracking, streaming responses, retries, dead-letter queues and document versioning.

---

## 2. Lesson 1 — LLM Fundamentals

**Goal:** not to learn how to use an LLM API, but to understand *what an LLM is doing underneath a RAG system*, so that retrieval, context windows, chunking and prompt construction later make sense.

### 2.1 What an LLM actually does

```text
Input text
    ↓
Convert text into tokens
    ↓
Convert tokens into numerical representations
    ↓
Neural network
    ↓
Predict probability of next token
    ↓
Choose a token
    ↓
Repeat
```

Example — given `The capital of India is`, the model conceptually computes:

```text
Delhi      → 0.94
Mumbai     → 0.02
Kolkata    → 0.01
Bengaluru  → 0.01
...
```

It picks `Delhi`, the sequence becomes `The capital of India is Delhi`, and it predicts the next token. This continues until the model decides the response is complete.

**The important idea:** an LLM is *not* doing "question → search database → find answer". It is doing "existing context → predict next token → predict next token → …". This distinction is extremely important for RAG.

### 2.2 Why does it know things then?

Because of training. Given training data such as:

```text
India is a country in South Asia.
The capital of India is New Delhi.
India has 28 states and 8 union territories.
```

Over billions/trillions of examples the model repeatedly learns to predict next tokens, learning statistical relationships between concepts:

```text
India
  ├── South Asia
  ├── New Delhi
  ├── Hindi
  ├── rupee
  ├── states
  └── ...
```

This is **not** the model storing a Wikipedia database. Its knowledge is encoded in the **parameters/weights** of the neural network.

### 2.3 Parameters

Models are advertised as `7B`, `70B`, `405B` parameters — `B` = billion. A parameter is essentially a learned numerical value inside the model:

```text
Model
  ├── parameter 1 = 0.182
  ├── parameter 2 = -0.732
  ├── parameter 3 = 0.019
  ├── parameter 4 = 1.234
  ...
  └── parameter N
```

Roughly, training works as:

```text
Training data
      ↓
Prediction
      ↓
Compare prediction with actual token
      ↓
Calculate error
      ↓
Adjust parameters
      ↓
Repeat billions of times
```

### 2.4 Tokens (first introduction)

LLMs do not process `"Hello, how are you?"` as characters or words in the normal sense. They process **tokens**. A tokenizer might represent it roughly as:

```text
["Hello", ",", " how", " are", " you", "?"]
```

A token can be a whole word, part of a word, punctuation, whitespace + word, or a special token. E.g. `unbelievable` might split into `un` / `believ` / `able`, while a common word may be one token.

**Why this matters for RAG:** RAG is heavily constrained by tokens. With a 128,000-token context window you cannot send an entire 500-page company document, so you must chunk → retrieve → fit useful context into the token budget. That is one of the fundamental reasons RAG exists.

### 2.5 Context window

The **context window** is the amount of tokenized information the model can consider in one request:

```text
┌──────────────────────────────────────┐
│          Context Window              │
│                                      │
│ System instructions                  │
│ Conversation history                 │
│ User question                        │
│ Retrieved RAG documents              │
│ Tool results                         │
│ ...                                  │
└──────────────────────────────────────┘
```

Example with a 32,000-token window:

```text
System prompt       2,000
Conversation         5,000
User question         200
Retrieved context   20,000
                    -------
                    27,200     ← still within the limit
```

If retrieval returns 50,000 tokens, it cannot all be sent — better retrieval / chunking / context management is needed.

### 2.6 Context ≠ Knowledge

This distinction is **very important**. Telling the model *"The company introduced a new leave policy called Policy X in August 2026"* puts that information into the **context**. It does **not** mean the model has permanently learned it.

```text
Model's trained knowledge
          +
Current context
          +
Current instructions
          ↓
       Response
```

RAG exploits this: information the model **did not have during training** can be placed into the context, letting it answer questions about private/company data.

### 2.7 Why RAG is needed

The LLM was not necessarily trained on the company's handbook, docs, FAQs or support tickets. RAG changes the situation:

```text
User: "What is our reimbursement policy?"
        ↓
Retrieve relevant company documents
        ↓
Context: "Employees can claim ..."
        ↓
LLM
        ↓
"According to the company policy..."
```

### 2.8 Hallucination

Ask *"What is the reimbursement limit in my company?"* and the model does not know — but an LLM is optimized to generate **plausible** text, so it may answer:

```text
The reimbursement limit is ₹50,000 per year.
```

even though that is completely made up. **This is a hallucination.**

RAG tries to supply authoritative context (retrieve policy → policy says ₹25,000 → LLM → answer ₹25,000).

**But RAG does not automatically eliminate hallucinations.** If retrieval returns the wrong document:

```text
Question → Wrong document → LLM → Wrong answer
```

This is why retrieval quality, grounding and evaluation matter later.

### 2.9 Temperature

Temperature controls how the model samples possible next tokens. Given `The sky is usually...` with candidate predictions `blue 0.85 / cloudy 0.08 / gray 0.04 / green 0.01`:

```text
Temperature ↓   →  More deterministic
                   More predictable
                   Less creative

Temperature ↑   →  More varied
                   More creative
                   Less predictable
```

For RAG-based factual question answering, high creativity is generally not needed.

### 2.10 System vs User messages

Modern LLM APIs distinguish between `system`, `user` and `assistant` messages. Plain version:

```text
SYSTEM: You are an HR assistant. Answer only using the provided company policy.
USER:   How many vacation days do I get?
```

With RAG:

```text
SYSTEM:  You are an HR assistant. Answer only using the provided context.
CONTEXT: Employees receive 24 annual leave days...
USER:    How many vacation days do I get?
```

Notice: **RAG isn't changing the model — it changes the context given to the model.**

### 2.11 Mental model to remember

```text
                  LLM
                   │
         ┌─────────▼─────────┐
         │   Context         │
         │ System prompt     │
         │ User question     │
         │ Conversation      │
         │ Retrieved data    │
         │ Tool results      │
         └─────────┬─────────┘
                   ▼
          Next-token prediction
                   ▼
               Answer
```

And RAG introduces:

```text
  User → Question → Retriever → Relevant documents → Context → LLM → Answer
```

### 2.12 Checkpoint

- Tokens connect almost everything in RAG: `documents → chunking → embeddings → retrieval → context → LLM → cost → latency`.

---

## 3. Lesson 2 — Tokens and Tokenization

More important for RAG than it initially looks.

### 3.1 What exactly is a token?

A **token is the basic unit of text that an LLM processes**. It is **not necessarily a word**.

```text
"I love Python"                 → ["I", " love", " Python"]        (3 tokens)
"unbelievable"                  → ["un", "believ", "able"]
"Hello, world!"                 → ["Hello", ",", " world", "!"]
```

The exact result depends on the tokenizer/model.

### 3.2 Why don't LLMs just use words?

Because the vocabulary would become unmanageable — `running / runs / ran / runner / runningly / ...`, plus names, URLs, programming code, misspellings, different languages, technical terms and newly invented words.

Modern tokenizers therefore use **subword tokenization**, so unfamiliar words can be represented by composing smaller pieces:

```text
authentication   → auth + entication
Django           → one token or multiple, depending on the tokenizer
```

This lets the model handle an enormous variety of text without a unique token per possible word.

### 3.3 The tokenization pipeline

```text
Human text → Tokenizer → Tokens → Token IDs → LLM
```

Example: `"Hello world"` → `["Hello", " world"]` → `[15496, 995]`. Those numbers are **token IDs**. The model does not receive English words directly — it receives numerical representations derived from these IDs.

### 3.4 Token ID vs embedding — an important distinction

```text
"Python" → Tokenizer → token ID = 12345
12345    → [0.12, -0.44, 0.81, ...]
```

So conceptually:

```text
Text → Tokenization → Token IDs → Embeddings / representations → Transformer
```

> The embedding used for semantic retrieval is **not** simply the token ID or the model's internal token embedding.

### 3.5 Why token count matters

With a 100,000-token context window:

```text
System prompt        2,000
Conversation         8,000
User question          100
Retrieved documents  60,000
                     ------
                     70,100     ← fine
```

But if retrieval returns 120,000 tokens of documents, it cannot all be sent. This is one reason RAG needs **chunking, retrieval, ranking, reranking and context selection**.

### 3.6 Chunking is directly related to tokens

A 100-page PDF should not become one giant embedding. Instead:

```text
PDF → Text → Chunks (400 tokens each) → Embedding → Vector DB
```

Then a question retrieves only the top 5 relevant chunks, and only those enter the LLM context.

### 3.7 Token budget

The context can be treated as a budget:

```text
System instructions       1,500
Conversation               4,000
User question               200
Retrieved context         20,000
Response reserve           6,300
                          -------
                           32,000
```

The **response reserve** matters: the context window is not just input — you also need room for the model's output.

```text
┌─────────────────────────────────┐
│       Context Window            │
│ System prompt                   │
│ Conversation                    │
│ User question                   │
│ Retrieved context               │
│ ─────────────────────────────── │
│ Space for generated answer      │
└─────────────────────────────────┘
```

A RAG system must therefore manage **both input and output token budgets**.

### 3.8 Tokens affect cost

```text
Cost = input tokens × input price + output tokens × output price
```

20,000 input tokens vs 5,000 input tokens — the second request can be substantially cheaper. So a bad RAG system that retrieves 30 irrelevant chunks is not just less accurate; it can also be **slower, more expensive, noisier and harder for the model to reason over**.

### 3.9 The "Lost in the Middle" problem

If you send `Chunk 1 … Chunk 20` and the relevant answer is buried in the middle, the model *technically has the information*, but its ability to use information varies by position in a long context.

```text
Relevant information
        ↓
┌───────────────────────┐
│ Beginning      ✓      │
│ Lots of information   │
│ Middle         ???    │
│ Lots of information   │
│ End            ✓      │
└───────────────────────┘
```

Therefore:

> **More retrieved context ≠ better RAG.** You want *high-quality* context, not *maximum* context.

### 3.10 Tokenization of programming code

```python
def authenticate_user(username, password):
    return verify_password(username, password)
```

The tokenizer may split identifiers and syntax into pieces such as `["def", " authenticate", "_user", "(", "username", ",", ...]`. Exact tokens depend on the tokenizer. This matters when building RAG over **source code, API documentation, SQL, JSON, YAML and logs** — a generic text chunker is not always ideal for those formats.

### 3.11 Token-level engineering decisions

Given a knowledge base of Order/Inventory service docs, Kafka docs and API docs, asking *"What happens when inventory reservation fails?"* forces decisions:

```text
How should I split these documents?
        ↓
How large should each chunk be?
        ↓
How much overlap?
        ↓
How many chunks should I retrieve?
        ↓
How many tokens can I put into the prompt?
```

These are **token-level engineering decisions**.

### 3.12 The subtle point — never memorize "1 word = 1 token"

That is wrong. Rough intuition for English:

```text
1 token ≈ ¾ of an English word
```

But this varies significantly. Common English text tokenizes efficiently, whereas long runs of repeated characters (`aaaaaaaaaaaaaaaaaaaa`), long technical identifiers (`someVeryLongTechnicalIdentifier`), or CJK text (`中文文本`) behave very differently.

**So when building serious systems, measure token counts using the tokenizer for the model you are actually using.**

### 3.13 The complete picture

```text
USER → Text → Tokenizer → Token IDs → LLM representations
     → Transformer → Next-token probabilities → Generated token → Repeat
```

RAG adds a parallel retrieval branch that feeds *context* into the tokenization stage.

---

## 4. Lesson 3 — Transformers and Attention

The key question:

> **How does the model determine which parts of the input are important when predicting the next token?**

That is where **attention** comes in.

### 4.1 The problem: words depend on other words

- *"The animal didn't cross the road because **it** was tired."* → `it` ≈ `animal`
- *"The animal didn't cross the road because **it** was too wide."* → `it` ≈ `road`

The model must understand relationships between different parts of the sentence. Attention provides the mechanism.

### 4.2 The basic idea of attention

Think of every token asking: **"Which other tokens should I pay attention to?"**

For `The animal didn't cross the road because it was tired`, the token `it` attends strongly to `animal` and less to `the`, `road`, `because`.

### 4.3 Attention is not a lookup rule

The model does **not** have a rule like `if token == "it": look_for_previous_noun()`. Attention is learned through neural network computations:

```text
Token representations → Attention (Q × K → score → probabilities) → Weighted information
```

### 4.4 Query, Key, Value

Every token produces three representations:

```text
Token
  ├── Query
  ├── Key
  └── Value
```

- **Query** ≈ "I'm looking for information relevant to me."
- **Keys** of other tokens ≈ "Here's the kind of information I contain."
- The model compares the Query against all Keys to produce attention scores.

### 4.5 Conceptual example

For `The cat sat on the mat because it was tired`, attention from the token `it` might look like:

| Token | Attention score |
|---|---|
| The | 0.02 |
| cat | 0.72 |
| sat | 0.03 |
| on | 0.01 |
| the | 0.01 |
| mat | 0.10 |
| because | 0.03 |
| it | 0.03 |
| was | 0.02 |
| tired | 0.03 |

The exact numbers are irrelevant. The idea is that `it` strongly attends to `cat`, so the model can incorporate information from `cat` when processing `it`.

### 4.6 Where Value comes in

Once attention decides *which* tokens matter, the model uses their **Values**:

```text
Query("it") → compare with Keys → attention weights → weighted Values → new representation for "it"
```

The (very simplified) formula:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

You don't need to memorise this yet, but you should recognise it.

### 4.7 Why "self"-attention?

Because tokens attend to **other tokens within the same sequence** — the sequence attends to itself.

```text
┌─────────────────────────────────────┐
│ I love Python because it is powerful│
│ ↑   ↕     ↕       ↕     ↕       ↕  │
└─────────────────────────────────────┘
```

### 4.8 Why attention matters for RAG

With a prompt containing several retrieved documents (e.g. Document A says 20 days annual leave, Document B says 24 days for employees in India, Document C says contractors get none), the LLM must connect the question's phrase "employees in India" to Document B. Attention allows tokens in the question and context to influence one another.

```text
Question → (attention) → Retrieved context → Relevant information → Generated answer
```

This is one reason RAG works.

### 4.9 Attention does NOT retrieve documents

A crucial distinction. There are **two separate concepts**:

- **Retrieval** — happens *outside* the LLM: `Question → Embedding → Vector DB → Relevant chunks`
- **Attention** — happens *inside* the transformer: `Question + retrieved chunks → Transformer → Attention → Answer`

```text
Vector search ≠ Attention
```

They solve different problems.

### 4.10 The full RAG pipeline, corrected

```text
USER QUESTION → Retriever → Relevant chunks
              → PROMPT (system instructions + retrieved context + user question)
              → Tokenization → Token representations
              → Transformer (Self-Attention → Feed Forward → …)
              → Next-token prediction → Answer
```

### 4.11 Multi-head attention

Transformers use multiple attention heads, each able to learn different relationships:

```text
                     Tokens
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
        Head 1       Head 2       Head 3
        syntax      entities     relationships
           └───────────┼───────────┘
                       ▼
               Combined representation
```

Possible learned patterns: `subject ↔ verb`, `pronoun ↔ noun`, `question ↔ relevant information`. **Do not interpret these as fixed rules** — actual head behaviour is far more complicated.

### 4.12 A transformer is not just attention

A transformer block contains several components:

```text
Input
  ↓
Self-Attention
  ↓
Add & Normalize
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

A large LLM stacks many such blocks:

```text
Input → Transformer Block 1 → … → Transformer Block N → Output probabilities
```

### 4.13 Why context length matters

With 100 tokens the model can attend across those tokens; with 100,000 there is potentially a huge amount of information it can process together. **But more context does not automatically mean better answers:**

```text
Question
   ↓
100 relevant tokens
+
99,900 irrelevant tokens
```

The model has access to everything, but now there is a lot of noise. This is why RAG is fundamentally a **retrieval problem** as much as an LLM problem.

### 4.14 Positional information

Attention alone does not give tokens their ordering — `Dog bites man.` and `Man bites dog.` contain the same words but mean opposite things. Transformers therefore add positional representations/encodings so that "A before B" is distinguishable from "B before A". Modern architectures implement this in different ways; the variants need not be memorised yet.

### 4.15 The key distinction to remember

> **Retrieval decides what information gets into the context.**
> **Attention determines how the model uses relationships among the information already in its context.**

This distinction becomes critical at the embeddings/vector-search stage, because embeddings are what make the retrieval side of RAG work.

---

## 5. Side Question — What is a Feed-Forward Network?

**Feed-Forward Network (FFN)** is the other major component inside a Transformer block, alongside self-attention.

> **Attention decides what information to look at; the feed-forward network processes and transforms that information.**

### 5.1 What the FFN does

Each token representation passes through a small neural network:

```text
Input vector → Linear transformation → Activation function → Another linear transformation → Output vector
```

Simplified:

```text
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂
```

No need to memorise it yet.

### 5.2 Why is it needed?

Attention establishes relationships (`cat ↔ sitting`, `cat ↔ is`, `sitting ↔ mat`), but the model still needs to **transform that information into a richer representation**. That is the FFN's job.

### 5.3 A useful backend analogy

```text
API Request → Serializer → Service Layer → Database
```

The serializer does not do the business logic; it transforms incoming data into a useful representation. Roughly:

```text
Attention → gathers/combines relevant information
FFN       → transforms/processes that information
```

Not an exact analogy, but useful mentally.

### 5.4 Real transformer block

```text
Input → Self-Attention → Residual + LayerNorm → FFN → Residual + LayerNorm → Output
```

repeated N times.

### 5.5 The one thing to remember

> **Self-attention mixes information between tokens. FFN transforms the information within each token's representation.**

For RAG, this helps you understand what happens **after** your retrieved chunks have been placed into the LLM's context.

---

## 6. Lesson 4 — Embeddings

### 6.1 What is an embedding?

An **embedding is a numerical representation of data that captures useful semantic information.**

```text
"How do I reset my password?"
        ↓
[0.021, -0.183, 0.721, 0.442, ..., -0.091]
```

Possibly 768 or 1536 dimensions, depending on the model.

It converts human language into numbers — but the crucial property is that the numbers are arranged so that **semantically similar text tends to have similar vectors**.

### 6.2 Why is that useful?

```text
A = "How do I reset my password?"
B = "I forgot my password. How can I change it?"
C = "What is the weather in Bengaluru?"
```

```text
A → [0.12, 0.81, 0.32, ...]
B → [0.14, 0.79, 0.35, ...]
C → [-0.72, 0.11, 0.91, ...]
```

A and B are semantically similar, therefore `similarity(A, B) → high` and `similarity(A, C) → low`. This gives us **semantic search**.

### 6.3 Traditional keyword search vs semantic search

Given documents:

```text
Document 1: "To change your password, visit Account Settings."
Document 2: "Your username can be modified from your profile."
Document 3: "Resetting forgotten credentials requires identity verification."
```

User asks: *"I forgot my password. How can I regain access?"*

**Keyword search** looks for matching words (`forgot`, `password`, `regain`, `access`) — but the documents say `forgotten`, `credentials`, `reset`. There may be no exact match.

**Semantic search** embeds the question and the documents and compares geometry:

```text
                   Question
                      │
                   Vector
                      │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Doc 1        Doc 2        Doc 3
      0.82          0.31          0.91
```

Document 3 is most semantically similar — even though it shares few words.

### 6.4 Embeddings as coordinates

With only 2 dimensions, similar concepts cluster: `reset password`, `forgot password`, `change credentials` group together, while `weather`, `football`, `pizza` sit elsewhere. Real embeddings have hundreds or thousands of dimensions, but the geometric idea is identical.

### 6.5 What "semantic" means

Semantic means **meaning**:

```text
"How can I change my password?"
"I forgot my credentials and need to regain access."
```

Different words, similar meaning. A good embedding model places them relatively close. This is the key advantage over simple keyword matching.

### 6.6 How an embedding model produces the vector

Conceptually:

```text
Text → Tokenizer → Transformer → Representation → Pooling / projection → Embedding vector
```

The exact architecture differs between embedding models — do not assume every embedding model follows this exact implementation.

### 6.7 Embedding model vs LLM

They are different things:

```text
LLM              → Question + Context → Answer              (generation)
Embedding model  → Question → Vector → Vector Database      (retrieval)
```

```text
                 USER
                  │
               Question
                  │
         ┌────────┴────────┐
         ▼                 │
  Embedding Model          │
         ▼                 │
    Query Vector           │
         ▼                 │
   Vector Database         │
         ▼                 │
Relevant Documents         │
         └────────┬────────┘
                  ▼
                LLM
                  ▼
               Answer
```

**This separation is fundamental.**

### 6.8 Comparing two embeddings

Given question embedding `Q` and document embedding `D`, similarity measures how closely their directions align:

```text
Question "How do I reset my password?"
  Document A "Password reset instructions..."   similarity = 0.91
  Document B "Today's weather forecast..."      similarity = 0.12
```

### 6.9 Why cosine similarity?

Imagine vectors as arrows. Small angle → high cosine similarity; large angle → lower similarity.

```text
cosine_similarity(A, B) = (A · B) / (||A|| ||B||)
```

No need to memorise the formula. The important concept:

> **We are comparing the geometry of vectors to estimate semantic similarity.**

(Cosine similarity, Euclidean distance and dot product were all listed in the Phase 2 roadmap; only cosine has been covered in depth so far.)

### 6.10 Embeddings + vector database = retrieval

With 100,000 chunks:

**Ingestion:** `Document → Chunk → Embedding → Vector → Vector DB`

**Query:** `Question → Embedding Model → Query Vector → Vector DB → Similarity Search → Top K chunks`

```text
Top 5:
Chunk 38291 → 0.94
Chunk 71922 → 0.91
Chunk 18273 → 0.88
Chunk 50182 → 0.84
Chunk 93012 → 0.81
```

Those chunks then go to the LLM. **That is basic RAG.**

### 6.11 Embeddings do not contain the original text

```text
Chunk: "Employees receive 24 days of annual leave."
   ↓
[0.12, -0.82, 0.33, ...]
```

The vector does **not** mean `0.12 = Employees`. It is not a dictionary — it is a **distributed representation**. Therefore you normally store both:

```text
┌─────────────────────────────────────┐
│ document_chunk                      │
├─────────────────────────────────────┤
│ id                                  │
│ content                             │
│ embedding                           │
│ metadata                            │
└─────────────────────────────────────┘
```

> The vector helps you **find** the chunk. The original text is what you eventually **give to the LLM**.

### 6.12 A common misconception

Do not think:

> "Embedding converts text into numbers so the computer understands English."

Too simplistic. More accurately:

> **An embedding model maps text into a high-dimensional vector space where certain semantic relationships can be represented geometrically.**

### 6.13 Why chunking matters to embeddings

Embedding an entire 500-page document produces one vector that represents *everything* and is not specific enough to answer "What is the password reset process?". Instead:

```text
500-page PDF → 10,000 chunks → 10,000 embeddings
```

Now retrieval can find the chunks specifically about password reset. This is why **chunking + embeddings + vector search** are tightly connected.

### 6.14 A subtle issue: similarity ≠ relevance

Question: *"How do I cancel an order?"*

```text
1. How to cancel an order       0.91
2. Order cancellation policy    0.89
3. How to create an order       0.87
4. Order status API             0.86
5. Order refund process         0.84
```

All are semantically related, but only some actually answer the question. That is why advanced RAG introduces `Retrieval → Reranking → Context selection`.

### 6.15 Embeddings in a backend architecture

```text
documents
─────────
id, title, source, metadata

document_chunks
─────────
id, document_id, content, embedding, metadata
```

```text
User Question → Embedding API → query_embedding
              → PostgreSQL + pgvector
              → ORDER BY vector similarity
              → Top 5 chunks
```

This is implementable without LangChain — and is exactly the recommended way to learn.

### 6.16 The complete mental model

> **We split documents into chunks, generate embeddings for those chunks, store them in a vector database, embed the user's question, retrieve semantically similar chunks, put those chunks into the LLM's context, and let the LLM generate an answer using that context.**

---

## 7. Clarification — We Store Both the Chunk Text and the Embedding

Question raised: *"but you told we store actual chunk also, to give to llm"* — confirmed correct.

```text
document_chunks
─────────────────────────────────────
id        content                         embedding
─────────────────────────────────────
1         "Password reset requires..."   [0.12, ...]
2         "Orders can be cancelled..."   [0.81, ...]
3         "Refunds are processed..."     [0.43, ...]
```

When a user asks *"How do I reset my password?"*:

1. Embed the question → `[0.14, 0.72, -0.31, ...]`
2. Search the **embedding column** by similarity → `Chunk 1 → 0.94 (relevant), Chunk 3 → 0.42, Chunk 2 → 0.21`
3. The database returns the matching **row** — including `content`
4. Take the **content**, not the vector, into the prompt
5. The LLM answers *"Go to Account Settings…"*

### Purpose of each stored field

| Data | Purpose |
|---|---|
| **Chunk text** | Give to the LLM |
| **Embedding** | Find relevant chunks |
| **Metadata** | Filter / identify chunks |

```text
DOCUMENT → CHUNK → ┬─ TEXT ────────────────┐
                   └─ EMBEDDING → VECTOR DB → similarity search ─┐
                                                                 ▼
                                                          RETRIEVED CHUNK → LLM → ANSWER
```

**The answer to "why not store only the embedding?":** the embedding lets us *locate* the relevant information, but the LLM needs the original text to actually use that information and generate the answer.

---

## 8. Lesson 5 — Vector Search and Vector Databases

The question this lesson answers:

> **How do we efficiently find the chunks whose embeddings are most similar to a user's question?**

### 8.1 The basic idea

```text
Chunk A: "To reset your password, go to Settings → Security → Reset Password."
Chunk B: "You can change your email address from your account profile."
Chunk C: "Password reset links expire after 15 minutes."
```

```text
Query: "How can I change my password?"  → [0.15, 0.81, -0.19, ...]
   ├── Chunk A = 0.96
   ├── Chunk B = 0.42
   └── Chunk C = 0.91
```

Retrieval returns Chunk A and Chunk C, which go to the LLM.

### 8.2 What exactly is vector search?

> **Given a query vector, find the vectors that are closest to it.**

Then rank them and usually ask for **Top-K** (e.g. Top-K = 3 → "give me the 3 most similar chunks").

### 8.3 How similarity is measured

**Cosine similarity** is extremely common for embeddings:

```text
             A · B
cosine = ─────────────
         |A| × |B|
```

- Very similar direction → `≈ 1`
- Opposite direction → `≈ -1`
- Unrelated / orthogonal → `≈ 0`

The useful range and interpretation depend on the embedding model and normalization.

### 8.4 Why not just use normal database search?

```sql
SELECT * FROM documents WHERE content LIKE '%password%';
```

That is **keyword search**. It finds `"password reset"` but not *"How do I regain access to my account?"* — there may be no word `password` at all.

```text
Keyword search → Words → matching words
Vector search  → Meaning → similar meaning
```

### 8.5 Where are vectors stored?

A **vector database** stores and searches high-dimensional vectors efficiently.

- PostgreSQL + pgvector
- Qdrant
- Milvus
- Weaviate
- Pinecone

**PostgreSQL + pgvector is the recommended starting point**, because it keeps the concepts close to a normal database.

### 8.6 PostgreSQL + pgvector

pgvector adds a vector column type to a table:

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
);
```

- `content` → the actual chunk
- `embedding` → vector representation

One row might hold `id: abc-123`, `content: "To reset your password, open Settings..."`, `embedding: [0.123, -0.452, 0.781, ...]`.

### 8.7 Searching with pgvector

```sql
SELECT id, content
FROM document_chunks
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

- `ORDER BY embedding <=> query_embedding` → rank stored vectors by distance from the query vector
- `LIMIT 5` → give me the top 5

This turns retrieval into:

```text
User Question → Embedding Model → Query Vector
              → PostgreSQL + pgvector → Similarity Search → Top 5 Chunks
```

### 8.8 Scaling — how can a database search millions of vectors?

With 10 documents you can compare against everything. With **10 million chunks** at 1536 dimensions each, comparing against every vector would be expensive. This is where **vector indexes** come in.

### 8.9 Vector indexes and HNSW

Vector databases use specialised indexes for fast similarity search. One important family:

**HNSW — Hierarchical Navigable Small World**

Mental model: instead of comparing the query against every vector, HNSW builds a **graph** that lets you navigate efficiently toward nearby vectors.

City analogy — without an index:

```text
"Find the nearest coffee shop"
You visit EVERY building in the city.
```

With a good navigation structure:

```text
Start → nearest major area → smaller area → street → coffee shop
```

That is the intuition behind approximate nearest-neighbour search.

### 8.10 ANN — Approximate Nearest Neighbor

Instead of guaranteeing the mathematically exact closest vectors, ANN allows finding vectors that are **extremely likely** to be among the closest, much faster.

```text
Exact search        →  Accuracy: highest      Speed: slower at huge scale
ANN search          →  Accuracy: very high    Speed: much faster
```

The real goal is `very good retrieval + very low latency`, not mathematically perfect NN.

### 8.11 HNSW intuition

```text
Level 2:   A ─────────────── H
Level 1:   A ─── C ─── E ─── H
Level 0:   A─B─C─D─E─F─G─H
```

Start at a higher level to move quickly toward the correct region, then descend into more detailed levels. (Intuition only — the real HNSW algorithm is more sophisticated.)

### 8.12 The complete RAG retrieval pipeline

```text
Ingestion:  PDF / Website / Markdown / Docs
              → Chunking
              → Chunk Text + Embedding + Metadata
              → PostgreSQL + pgvector

Query:      User question → Embedding Model → Query Vector
              → pgvector similarity search → Top K chunks
              → Prompt → LLM → Answer
```

### 8.13 Three different things to keep separate

| Component | Question it answers | Transform |
|---|---|---|
| **Embedding model** | "How should I represent this text as a vector?" | text → vector |
| **Vector database** | "Which stored vectors are closest to this query vector?" | query vector → similar vectors |
| **LLM** | "Given this context, what should I say?" | context + question → answer |

```text
Embedding Model → Vector DB → Retrieved Text → LLM
```

### 8.14 Metadata becomes important

Real RAG systems store more than content and embedding:

```text
id
document_id
content
embedding
page_number
section
source
created_at
tenant_id
```

Example row: `id: chunk-123`, `document: employee-handbook.pdf`, `page: 42`, `section: Password Management`, `tenant: company-A`, `content: "Password must be changed every…"`, `embedding: [...]`.

Why? Because you may want to *"search only documents belonging to company-A"*, making retrieval:

```text
Vector similarity  +  Metadata filtering
```

This becomes extremely important in production RAG.

### 8.15 Vector search isn't the whole retrieval story

```text
Basic RAG:                       Production:
Query                             Query
  ↓                                 ↓
Embedding                          Query processing
  ↓                                 ↓
Vector search                      Hybrid retrieval
  ↓                                 ↓
Top 5                              Vector search + Keyword search
                                   Metadata filtering
                                   Reranking
                                   Top relevant chunks
                                   LLM
```

### 8.16 One thing you should NOT conclude

Do not think:

> "Vector similarity = understanding."

For *"What is the refund policy?"*, vector search might return refund policy, purchase policy, cancellation policy, payment policy, customer support — all semantically related, but perhaps only the first actually answers the question. Hence reranking and retrieval evaluation.

### 8.17 The mental model — RAG as a librarian

- **Embedding model** — turns your question into a conceptual location.
- **Vector database** — finds books/pages located near that conceptual location.
- **LLM** — reads those retrieved pages and answers.

### 8.18 Lesson 5 checkpoint

1. **What is vector search?** Finding vectors similar/close to a query vector.
2. **What is Top-K?** Return the K highest-ranked results.
3. **Why vector search instead of only keyword search?** It retrieves on semantic similarity rather than exact word matches.
4. **What is a vector database?** A system optimised for storing and searching vector representations.
5. **Why do we need HNSW/ANN?** To make nearest-neighbour search efficient at large scale.
6. **What does pgvector give PostgreSQL?** Vector data types, similarity/distance operators, and vector indexes/search.
7. **What does the vector DB return?** Usually `chunk_id`, `content`, `metadata`, `similarity/distance score`.
8. **What does the LLM receive?** User question + retrieved chunk text + instructions — **not just the embeddings**.

---

## 9. Lesson 6 — The RAG Pipeline (Conceptually)

### 9.1 What is being built

A small **documentation Q&A system** over `docs/` containing `authentication.txt`, `payments.txt`, `orders.txt`. A user asks *"How do I reset my password?"* and the system retrieves the right chunks, builds a prompt, and answers.

### 9.2 Two separate pipelines

This distinction is extremely important.

**Pipeline A — Ingestion** (when documents enter the system)

```text
Document → Read → Split into chunks → Create embeddings
        → Store: chunk text + embedding + metadata
```

**Pipeline B — Query** (every time the user asks something)

```text
User question → Create embedding → Vector search → Retrieve chunks
             → Send chunks + question to LLM → Answer
```

```text
                RAG
                 │
       ┌─────────┴─────────┐
   INGESTION              QUERY
   Documents              Question
      Chunks                 ↓
   Embeddings             Embedding
      ↓                       ↓
   Vector DB             Vector Search
                              ↓
                       Retrieved Chunks
                              ↓
                             LLM
                              ↓
                           Answer
```

### 9.3 The knowledge base

A tiny `docs/authentication.txt` describing password reset: reset from Settings, navigate Settings → Security → Reset Password, enter registered email, link emailed, link expires after 15 minutes, check spam folder if not received.

### 9.4 Chunking in the pipeline

Documents are split so retrieval works on focused pieces:

```text
Document → Chunk 1, Chunk 2, Chunk 3, …
```

- If a chunk is enormous (e.g. 5,000 words) you may retrieve lots of irrelevant information.
- If it is tiny (3 words) you may lose the context needed to answer.

**Chunking is one of the most important parts of RAG.**

### 9.5 Embeddings and storage

Each chunk gets its own vector (e.g. 1536 numbers):

| id | content | embedding |
|---|---|---|
| 1 | Users can reset… | `[...1536 values...]` |
| 2 | Open Settings… | `[...1536 values...]` |
| 3 | Reset link expires… | `[...1536 values...]` |

> **Embedding is for retrieval. Content is for the LLM.**

### 9.6 Query flow

The question is never used to search the raw sentence — it is embedded first:

```text
Query → Chunk 1 → 0.72
        Chunk 2 → 0.94
        Chunk 3 → 0.61
```

With Top-K = 2, retrieval returns Chunk 2 and Chunk 1.

### 9.7 Building the context

Take the **actual text** from the retrieved chunks — never the vectors — and assemble a context block.

### 9.8 Giving context to the LLM

```text
System:  Answer the user's question using the provided context.
         If the answer isn't present in the context, say you don't know.
Context: Users can reset their password from the Settings page…
         Open Settings, select Security, and click Reset Password…
Question: Where can I reset my password?
```

LLM output: *"You can reset your password from Settings → Security → Reset Password."*

**That is RAG.**

### 9.9 The complete system

```text
INGESTION:  Documents → Chunking → Embedding Model
              → Vector Store (text + embedding + metadata)

QUERY:      User Question → Embedding Model → Query Embedding
              → Vector Search → Top-K Chunks → Context + Query → LLM → Answer
```

### 9.10 Realistic scale

1,000 documents × 100 chunks each = **100,000 chunks**, 100,000 embeddings, 100,000 chunk texts and metadata sets. A query returns ranked results and typically only the top 3 go to the LLM.

### 9.11 Two different models in play

| Model | Transform | Purpose |
|---|---|---|
| **Embedding model** | Text → Vector | retrieval |
| **Generative LLM** | Prompt → Answer | generation |

They do not have to be the same model.

### 9.12 Does the LLM search the vector DB?

**Normally no.** The **application orchestrates** the process: embed the question, run vector search, build the prompt from retrieved chunks, then call the LLM.

The LLM doesn't magically know *"go search PostgreSQL."* Your application provides the retrieved context.

> This distinction becomes **very important** when learning agents and tool calling.

### 9.13 RAG is an application architecture

RAG is not "an AI model" — it is a pipeline combining documents/data, chunking, embeddings, retrieval, context construction and LLM generation, plus often metadata filtering, reranking, citations and evaluation. **The LLM is only one component.**

### 9.14 Intended evolution of the learning project

| Version | Composition | Purpose |
|---|---|---|
| **V1** | Python + embedding API + in-memory vectors + LLM API | Understand the algorithm |
| **V2** | Python + PostgreSQL + pgvector + embedding model + LLM | A real RAG system |
| **V3** | FastAPI + PostgreSQL/pgvector + ingestion API + query API | An actual backend application |
| **V4** | + chunk metadata, hybrid search, reranking, conversation history, citations, evaluation | Quality improvements |
| **V5** | + LangChain, LangGraph, agents, tools | Understand *why* frameworks exist |

### 9.15 Lesson 6 mental model key distinction

> **Vector search finds the information. The LLM uses that information to generate the answer.**

---

## 10. Lesson 7 — Chunking Strategies

> **How do we divide a large document into chunks that are useful for retrieval?**

You can have a great LLM and a great embedding model and still get poor retrieval if documents are split badly.

### 10.1 Why chunk at all?

A 100-page PDF (100 pages, 50,000 words) must not become one giant embedding — a single vector representing that much information is not useful for precise retrieval. Instead the document is split and each chunk gets its own embedding.

### 10.2 The simplest method — fixed-size chunking

Split into units of a fixed size (e.g. 500 characters), producing `characters 1–500`, `501–1000`, `1001–1500`. Simple, but there is a problem.

### 10.3 The boundary problem

```text
Original:
The refund policy applies to all purchases.
Customers can request a refund within 30 days of purchase.

Blind split:
Chunk 1: "The refund policy applies to all purchases. Customers can"
Chunk 2: "request a refund within 30 days of purchase."
```

Chunk 2 says *"request a refund…"* but lacks *"Customers can…"*; the subject lives in the previous chunk. **Meaning has been broken across chunks.**

### 10.4 Chunk overlap

Instead of non-overlapping windows, neighbouring chunks share a tail:

```text
chunk_size = 500
overlap    = 100

Chunk 1 → characters 0–499
Chunk 2 → characters 400–899
Chunk 3 → characters 800–1299
```

So 100 characters are repeated.

### 10.5 Why overlap helps

When an important relationship straddles a boundary (`"...customers who purchased the product"` + `"...within the last 30 days can request a refund"`), overlap makes the important concept appear **in both chunks**.

### 10.6 But overlap isn't magic

"Let's use 50% overlap and solve everything" is not right. 100,000 words with huge overlap could produce 150,000+ words stored, meaning more embeddings, more storage, more retrieval candidates, more duplicated context and potentially higher LLM costs. Overlap should be used thoughtfully.

### 10.7 Character vs word vs token chunking

`chunk_size = 1000` is ambiguous — it could mean 1000 characters, 1000 words, or 1000 tokens. For LLM applications **token-based chunking** is more meaningful because context limits and costs are measured in tokens. A string like `"Hello, how are you?"` does not correspond to exactly 4 words → 4 tokens.

> **Tokens are not the same thing as words.**

### 10.8 Semantic boundaries are better

Given a document with `# Authentication`, `# Payments`, `# Shipping` headings, a bad chunker produces a chunk mixing the tail of the Authentication section with the start of Payments. A better chunker respects document structure and keeps each section whole — this is why **document-aware chunking** is valuable.

### 10.9 A common hierarchy

```text
Document
   ├── Title
   ├── Section
   │     ├── Paragraph
   │     └── Paragraph
   ├── Section
   └── Section
```

Rather than blindly splitting characters, split along:

```text
Document → Sections → Paragraphs → Sentences → Chunk size limit
```

giving the chunker increasingly fine-grained boundaries.

### 10.10 Recursive chunking

```text
Try splitting by:
  paragraph
      ↓ if chunk too large
  sentence
      ↓ if still too large
  word
      ↓ if still too large
  character
```

This is far more sensible than immediately cutting every 500 characters.

### 10.11 Chunk size — how big should a chunk be?

There is **no universal answer**. You may see 200 / 500 / 800 / 1000 / 1500 tokens, but do not memorise *"500 is the correct chunk size."*

| | Advantages | Potential problems |
|---|---|---|
| **Small chunks** | More precise retrieval, less irrelevant information | Context can be lost, important information may be split |
| **Large chunks** | More context, relationships between concepts preserved | More irrelevant information, larger prompts, less precise retrieval |

### 10.12 Think about retrieval, not just storage

Don't ask *"What chunk size is standard?"* Ask:

> **"What unit of information would I want returned if a user asked a question about this document?"**

Example: indexing API documentation, you'd want `GET /users/{id}` with Description, Parameters, Response, Example and Errors to stay together — so that *"What does GET /users/{id} return?"* is fully answerable from the retrieved result.

### 10.13 Different documents need different chunking

| Document type | Good boundaries |
|---|---|
| **Markdown documentation** | headings, sections, paragraphs |
| **Legal documents** | sections, subsections, clauses |
| **Source code** | file → class → method/function → logical blocks (never character ranges) |
| **Database documentation** | table → columns → relationships → examples |
| **PDFs** | page → heading → paragraph → table |

**Chunking is data-dependent.**

### 10.14 Metadata can preserve lost context

```json
{
  "document": "employee_handbook.pdf",
  "page": 42,
  "section": "Password Management"
}
```

Chunk text: `"Passwords must be changed every 90 days."`

Without metadata this chunk is isolated. With metadata you can present:

```text
Document: Employee Handbook
Section: Password Management
Page: 42

Passwords must be changed every 90 days.
```

Context has been preserved.

### 10.15 Contextual chunking

Given `# Refund Policy` → `Customers can request a refund within 30 days.`, the raw chunk could be enriched with its parent context:

```text
Document: Store Policy
Section: Refund Policy

Customers can request a refund within 30 days.
```

Then embed the **enriched** representation. This improves retrieval because the embedding carries more context — but the original text should still be preserved separately for generation.

### 10.16 Parent-child retrieval

```text
Parent document
      ├── Child chunk 1
      ├── Child chunk 2
      ├── Child chunk 3
      └── Child chunk 4
```

Embed the small child chunks for precision:

```text
Query → Child vector search → Child chunk 3 → Parent section → LLM
```

Instead of sending only the tiny matched chunk to the LLM, return its larger parent section. This gives:

> **Precise retrieval + richer context**

### 10.17 What if a chunk contains multiple topics?

A chunk holding password reset, user creation and invoice generation produces **one embedding representing three unrelated concepts**. A query about creating users might retrieve it — technically correct but inefficient. Splitting into three topical chunks improves semantic precision.

### 10.18 The real objective of chunking

> **A good chunk is a self-contained unit of information that can answer, or substantially contribute to answering, a user's question.**

Not *"a chunk must contain exactly 500 tokens."* Token size is just a constraint — **meaning is the goal.**

### 10.19 Recommended first chunking approach

```text
Step 1 — Split by paragraphs
Step 2 — Combine paragraphs until target token size
Step 3 — Use small overlap where appropriate
Step 4 — Store metadata
```

### 10.20 The full ingestion pipeline, expanded

```text
DOCUMENT → Document Parser → Structure (headings | paragraphs)
         → Chunking → Chunk + Metadata → Embedding Model
         → PostgreSQL + pgvector
```

### 10.21 Chunk before embedding

Not:

```text
Document → Embedding → Chunk
```

But:

```text
Document → Chunk → Embedding each chunk
```

Because we want `Chunk 1 → Embedding 1`, `Chunk 2 → Embedding 2`, `Chunk 3 → Embedding 3`. This lets vector search identify **which specific part of the document** is relevant.

### 10.22 Lesson 7 summary

| Concept | Purpose |
|---|---|
| Chunking | Split documents into retrievable units |
| Chunk size | Controls how much information each chunk contains |
| Overlap | Preserves context across boundaries |
| Token chunking | Uses LLM-relevant units |
| Recursive chunking | Splits using increasingly fine boundaries |
| Metadata | Preserves document/section/page context |
| Parent-child retrieval | Precise search + richer context |
| Semantic chunking | Keeps related information together |

The core principle:

```text
Bad chunking → Bad embeddings → Bad retrieval → Bad context → Bad answer
```

> **RAG quality starts before the vector database.**

---

## 11. Lesson 8 — Embedding Models in Practice

Practical questions this lesson answers: what creates these vectors, which model to use, why dimensions differ, can we switch models later, local vs API, and how embedding quality relates to RAG quality.

### 11.1 What is an embedding model?

A model specifically trained to convert things like text into vectors useful for comparing meaning.

```text
"How do I reset my password?"        → nearby vector
"Where can I change my password?"    → nearby vector
"How do I track my package?"         → farther vector
```

```text
Similar meaning → Similar vectors
```

is the fundamental property wanted.

### 11.2 Embedding models are not chat models

```text
LLM             → Question → GPT/Claude/Gemini → Answer   (writer / reasoner)
Embedding model → Text → Vector                            (semantic coordinate generator)
```

An embedding model does not answer *"How do I reset my password?"*; it returns a representation of the text.

### 11.3 Why not use the LLM itself to create embeddings?

Modern generative models internally have vector representations, but application-level embedding models are specifically trained to produce vectors useful for:

- semantic search
- clustering
- similarity
- retrieval
- classification
- recommendations

So the RAG architecture usually has an embedding model for retrieval and a chat LLM for generation.

### 11.4 Embedding dimensions

Common sizes: `384`, `768`, `1024`, `1536`, `3072`. The dimension is simply the length of the vector.

### 11.5 Is a larger dimension always better?

**No.** This is a common misconception. `3072 > 1536 > 768` does **not** mean 3072 must be better. Embedding quality depends on the **model and training**, not simply the number of dimensions. A smaller, well-trained embedding model can outperform a larger one for a particular retrieval task.

```text
More numbers ≠ automatically more useful information
```

### 11.6 Why dimension matters then — storage

```text
1 million vectors × 1536 dimensions × 4 bytes (32-bit float)
   ≈ 6 KB/vector
   ≈ 6 GB total
```

That is **before** database/index overhead. Vector dimensions matter significantly at scale.

### 11.7 Dimension also affects search

More dimensions mean more computation, more memory and larger indexes. So you don't choose *"the largest vector possible"* — you choose a model appropriate for your retrieval requirements and infrastructure.

### 11.8 A crucial rule — one vector space

If documents were indexed with **Embedding Model A** and queries are embedded with **Embedding Model B**, comparing them is generally **invalid**, because the vectors live in different embedding spaces.

### 11.9 Coordinate-system analogy

```text
Map A:  Delhi → (100, 200)      Mumbai → (500, 700)
Map B:  Delhi → (0.2, 0.8)      Mumbai → (0.9, 0.1)
```

You cannot meaningfully compare `(100, 200)` with `(0.2, 0.8)` just because both are coordinates. The same applies to embeddings.

### 11.10 Changing embedding models requires re-indexing

```text
Documents → Chunks → Model B → New embeddings → Rebuild vector index
```

You cannot simply change the query embedding while keeping old document embeddings.

### 11.11 Why this matters in production

For 10 million chunks, changing the embedding model is **not** a one-line change — it is an **index migration**, requiring thought about: old embeddings, new embeddings, storage, re-indexing, index creation, downtime / dual indexes, validation and rollback.

This is one reason embedding-model selection matters early.

### 11.12 Local vs API embedding models

**Option A — API**

```text
Your server → Embedding API → Vector
```

- Advantages: easy to implement, no model hosting, usually strong models, scaling handled by the provider
- Disadvantages: API cost, network latency, external dependency, data leaves your infrastructure

**Option B — Local model**

```text
Your server → Local embedding model → Vector
```

- Advantages: no per-request API cost, data stays in your infrastructure, control over model/version, works offline
- Disadvantages: CPU/GPU requirements, model management, deployment complexity, potentially weaker performance depending on model

### 11.13 What to use for learning

Use an **API-based** embedding model first. Because the goal is to understand `chunk → embedding → vector database → retrieval → LLM`, we do not want to simultaneously debug CUDA, model loading, quantization, GPU memory and PyTorch. Once the pipeline works, switching to a local embedding model is much easier.

### 11.14 One dimension does not mean one concept

Do not imagine `dimension 1 → password`, `dimension 2 → payment`. Information is **distributed across many dimensions**; the overall vector relationship matters.

```text
"reset password"   [ .12, .83, -.21, .44, .08, ... ]
"change password"  [ .11, .80, -.18, .47, .06, ... ]
```

### 11.15 Similarity scores are not probabilities

| Chunk | Score |
|---|---|
| "Password reset instructions…" | 0.94 |
| "Change your account password…" | 0.92 |
| "Update your email address…" | 0.61 |
| "Shipping information…" | 0.12 |

`0.94` does **not** mean 94% correct. A similarity score is a similarity/distance measure whose scale depends on the metric and the embedding model. Do not build rules like `0.90 = definitely correct` without evaluating your specific system.

### 11.16 Embedding quality directly affects RAG

A good embedding model retrieves the correct chunk; a weaker or poorly matched one retrieves generic neighbours instead. Since the LLM cannot reliably answer using information it never received:

```text
Bad retrieval → Bad context → LLM has limited ability to recover
```

> **Retrieval quality is often more important than simply choosing a more powerful generation model.**

### 11.17 Embedding models need evaluation

Don't choose a model only because "everyone uses it". Build a test set:

```text
Question 1 → expected chunk IDs
Question 2 → expected chunk IDs
...
Question 100 → expected chunk IDs
```

Run retrieval and measure: did the correct chunk appear in **Top 1? Top 5? Top 10?** These give retrieval metrics (studied later).

### 11.18 Dense, sparse and hybrid retrieval

**Dense retrieval** — text is represented as dense numerical vectors and searched by similarity. This is what has been learned so far.

**Sparse retrieval** — term-based; the classic example is **BM25**, closer to keyword search.

**Hybrid search** — combining the two. Extremely useful in RAG.

### 11.19 Practical example — why hybrid matters

```text
API endpoint: POST /v1/orders/{order_id}/cancel
```

- *"Which endpoint cancels an order?"* → semantic search works well
- *"What is `/v1/orders/{order_id}/cancel`?"* → exact keyword/token matching is extremely useful

```text
Semantic search  +  Keyword search  →  Hybrid retrieval
```

### 11.20 Updating a document

```text
Old: Refunds take 7 days.
New: Refunds take 5 business days.
```

Correct handling:

```text
Updated document → Re-chunk affected content
                 → Re-embed affected chunks
                 → Update vector database
```

You don't need to re-embed the whole corpus. This leads to the production concept:

> **Incremental indexing** — only changed documents/chunks are reprocessed.

### 11.21 The target production architecture

```text
DOCUMENT INGESTION
   Document Parser → Chunker → Embedding Model → PostgreSQL/pgvector
   (stores content + vector)

RETRIEVAL
   User Query → Query Embedding
              → Vector + Keyword Search
              → Reranking
              → Context Selection
              → LLM
              → Answer
```

### 11.22 Lesson 8 takeaways

- **Embedding model**: text → vector
- **Embedding dimension**: length of vector
- **Bigger dimension ≠ automatically better** — model quality matters
- **Same embedding space**: document embeddings + query embedding must come from compatible models
- **Changing embedding models** usually means re-embed documents + rebuild/update the vector index
- **Embedding model ≠ LLM**: retrieval vs generation
- **Dense retrieval**: semantic vectors + similarity search
- **Hybrid retrieval**: dense/vector search + sparse/keyword search

---

## 12. Lesson 9 — Retrieval: Top-K, Filtering, Hybrid Search, Reranking

Core of RAG quality. The problem: **vector search doesn't always return exactly the information you need.**

### 12.1 What exactly is retrieval?

Retrieval is the process of deciding:

> **Which pieces of information should be given to the LLM?**

With 100,000 chunks, you don't want all of them going to the LLM — you want retrieval to reduce them to ~5. So retrieval acts as a **filter between your knowledge base and your LLM**.

### 12.2 Top-K retrieval

If similarity results are `Refund policy 0.94 / Payment cancellation 0.91 / Refund processing time 0.89 / Payment methods 0.71 / Shipping policy 0.32 / Product catalog 0.10`, then `K = 3` returns the first three.

### 12.3 Why not just retrieve Top-1?

Sometimes the answer requires two chunks:

```text
Chunk A: Refunds can be requested within 30 days.
Chunk B: Approved refunds are processed within 5 business days.
Question: "How does the refund process work?"   → needs BOTH
```

Top-1 could miss part of the answer; Top-5 gives a better chance of capturing all necessary information.

### 12.4 But why not retrieve Top-100?

```text
Top 5:   5 relevant chunks
Top 100: 10 relevant, 90 irrelevant
```

The LLM must process a lot of noise, causing larger prompts, higher cost, higher latency, more distracting information and weaker answers.

```text
Too few chunks → Missing information
Too many chunks → Too much noise
```

### 12.5 Top-K is not a magic number

Do not memorise `Top-K = 5`. There is no universal correct K. You should eventually test `K = 3 / 5 / 10 / 20` against an evaluation dataset and measure which gives better retrieval quality — this is **retrieval tuning**.

### 12.6 Similarity threshold

An alternative/complement to Top-K: set `minimum_similarity = 0.80` and keep only results at or above it. This is called a **similarity threshold**.

### 12.7 Top-K vs threshold

| | Meaning | Behaviour |
|---|---|---|
| **Top-K** | "Always give me the best K results." | returns K even if weak |
| **Threshold** | "Give me results only if sufficiently similar." | may return fewer, or none |

Combine them:

```text
Top 10 → Remove results below threshold → Remaining chunks
```

Be careful: similarity scores are not universal probabilities, so thresholds must be tuned for your model and data.

### 12.8 The "no relevant answer" problem

```text
Knowledge base: Password docs, Payment docs, Shipping docs
Question:      "Who won yesterday's cricket match?"
```

Vector search may still return `Password documentation → 0.31, Payment documentation → 0.28, Shipping documentation → 0.24`. The system must recognise:

> "None of these are actually relevant."

Hence retrieval systems use:

```text
similarity threshold + reranking + LLM-level grounding instructions
```

rather than assuming *"Top-5 results must contain the answer."*

### 12.9 Metadata filtering

With three companies × 10,000 chunks each, a user from Company B should not search all 30,000 chunks. Filter by `tenant_id = "company_b"` and combine:

```text
Vector Search + tenant_id = company_b
```

### 12.10 Realistic chunk table

```text
id
document_id
content
embedding
tenant_id
source
page
section
created_at
```

Example:

```text
id:          chunk-982
document_id: doc-12
tenant_id:   company-b
source:      employee-handbook.pdf
page:        42
section:     Password Management
content:     Passwords expire every 90 days.
embedding:   [...]
```

Retrieval becomes *"find similar vectors WHERE tenant_id = 'company-b'"*.

### 12.11 Filtering BEFORE vector search

```text
30,000 chunks → Metadata filter → 10,000 chunks → Vector search → Top 10
```

Useful because you search a smaller relevant population. Note that depending on the database/index and query structure, filtering can interact with vector-index performance, so production implementations must consider the database's filtering behaviour.

### 12.12 Other useful metadata filters

| Filter type | Example |
|---|---|
| Document | `document_id = X` |
| User | `user_id = X` |
| Tenant | `tenant_id = X` |
| Date | `created_at >= …` |
| Document type | `document_type = "api_docs"` |
| Version | `version = "v2"` |

```text
Retrieval = Semantic similarity + Metadata constraints
```

This is one reason metadata matters so much during ingestion.

### 12.13 The problem with pure vector search

```text
POST /users
GET /users/{id}
DELETE /users/{id}
POST /users/{id}/reset-password
```

Question: *"What is the endpoint `/users/{id}/reset-password`?"* — an **exact identifier**. Vector search may understand the concept, but keyword search is particularly strong for exact strings.

### 12.14 What is hybrid search?

```text
Dense retrieval   → Question → Vector    → Semantic similarity
Sparse retrieval  → Question → Terms    → BM25 or similar
```

Then combine their results.

### 12.15 Why combine them?

- **Question A** — *"How can I regain access to my account?"* → semantic search useful, because the documentation says `reset password`
- **Question B** — *"What does `/v1/orders/{id}/cancel` do?"* → keyword search useful, because the exact API path matters

Semantic + keyword covers more query types.

### 12.16 BM25

BM25 asks: **how important are the query terms in this document?** It considers:

- term frequency
- how common the term is across documents (inverse document frequency)
- document length

So `"password reset"` strongly matches a chunk saying `"To reset your password…"`. No need to memorise the derivation yet.

### 12.17 Dense vs sparse mental model

| | Dense | Sparse |
|---|---|---|
| Representation | Vector | Terms |
| Captures | Meaning | Exact words |
| Good for | Semantic queries | Exact matches |
| Example | "regain account access" | `/users/{id}` |
| Typical method | Embeddings | BM25 |

Neither is universally better; they solve different retrieval problems.

### 12.18 The problem when combining results

```text
Vector search:  Chunk A → 0.94,  Chunk B → 0.90,  Chunk C → 0.86
BM25:          Chunk C → 12.4,  Chunk D → 11.7,  Chunk A → 10.9
```

Chunks A and C appear in both — potentially strong candidates. But:

> The scores aren't directly comparable.

You cannot simply do `0.94 + 12.4`, because the scores come from different scoring systems.

### 12.19 Reciprocal Rank Fusion (RRF)

Instead of comparing raw scores, RRF cares about **rank positions**:

```text
Vector ranking:   1. A   2. B   3. C   4. D
BM25 ranking:     1. C   2. A   3. D   4. E

A → rank 1 + rank 2
C → rank 3 + rank 1
D → rank 4 + rank 3
```

Documents appearing highly in multiple retrieval methods become strong candidates.

> **RRF combines rankings rather than directly comparing incompatible scores.**

No need to memorise the formula yet.

### 12.20 The retrieval pipeline grows

```text
Earlier:  Question → Embedding → Vector Search → Top-K → LLM

Now:     Question ─┬─ Vector Search ─┐
                    └─ BM25 Search ──┘
                            ↓
                    Combine Results
                            ↓
                    Metadata Filtering
                            ↓
                        Top-K
                            ↓
                        LLM
```

### 12.21 Reranking

If retrieval returns top 20 chunks — some relevant, some only vaguely related — a second model can **rerank** them:

```text
Query → Initial retrieval → 20 candidates → Reranker → Top 5 → LLM
```

### 12.22 Why not use the reranker for everything?

Reranking is more computationally expensive than initial retrieval, so a two-stage strategy is used:

```text
Stage 1 — Fast retrieval:      1,000,000  →  Top 50
Stage 2 — Accurate reranking:         50  →  Top 5
                                          →  LLM
```

This is called **retrieve → rerank**.

### 12.23 Why is reranking different from embedding similarity?

- Embedding retrieval asks approximately: *"Are these texts semantically close?"*
- A reranker evaluates: *"Given this specific query, how relevant is this particular passage?"*

```text
Embedding search → broad candidate retrieval
Reranker         → fine-grained relevance
```

### 12.24 The production-style retrieval architecture

```text
                         USER QUERY
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Query Embedding                  Query Terms
              │                             │
              ▼                             ▼
        Vector Search                    BM25
              └──────────────┬──────────────┘
                             ▼
                       Result Fusion
                             ▼
                    Metadata Filtering
                             ▼
                       Top 20–50
                             ▼
                         Reranker
                             ▼
                        Top 3–5
                             ▼
                      Context Builder
                             ▼
                            LLM
                             ▼
                          Answer
```

### 12.25 Retrieval ≠ generation

| | Question |
|---|---|
| **Retrieval** | "What information should I give the LLM?" |
| **Generation** | "What should the LLM say using that information?" |

If the answer is wrong there are at least two major possibilities:

```text
Wrong answer
    ├── Retrieval problem  → Wrong context
    └── Generation problem → Bad use of correct context
```

This distinction becomes crucial during RAG evaluation.

### 12.26 Worked example

Knowledge base: refund policy, payment configuration, shipping policy, API documentation.
Question: *"What is the API endpoint for cancelling an order?"*

- **Vector search** may retrieve payment configuration, order cancellation API, refund policy
- **BM25** (query contains "API", "endpoint", "cancelling", "order") strongly retrieves `POST /orders/{id}/cancel`
- **Fusion** → candidate set
- **Reranker** identifies `POST /orders/{id}/cancel` as most relevant
- **LLM** receives that chunk and answers

### 12.27 What to implement first

| Version | Add | Goal |
|---|---|---|
| **V1** | Vector Search + Top-K | basic loop |
| **V2** | Metadata filtering | scoped, safe retrieval |
| **V3** | BM25 + Vector Search | hybrid |
| **V4** | Reranking | better final context |
| **V5** | Evaluation | prove each component helps |

### 12.28 Lesson 9 mental model

> **Retrieval is not simply "find the nearest vectors."** It is the process of finding the **smallest useful set of evidence** for the LLM.

```text
Huge Knowledge Base → Candidate Retrieval → Filtering → Ranking
                   → Relevant Evidence → LLM
```

Progression learned:

```text
Vector Search → Top-K → Metadata Filtering → Hybrid Search
             → Reranking → Context Selection → LLM
```

---

## 13. Clarification — What Exactly Is a Reranker?

> A **reranker is a model that re-evaluates the relevance of documents retrieved by the initial search.** It is *not* usually another full search through the entire database — think of it as a second, more precise stage of retrieval.

### 13.1 How reranking works

```text
Step 1 — Initial search (vector similarity, BM25, or hybrid)
         1,000,000 chunks → 50 candidate chunks

Step 2 — Reranker model evaluates how relevant each candidate is
         to the user's actual question
         50 candidates → ranked by relevance → top 5 chunks

Step 3 — LLM generation receives the question and selected chunks
```

### 13.2 What kind of model is a reranker?

A common type is a **cross-encoder model**. Unlike a typical embedding model — which independently converts query and document into vectors — a cross-encoder takes them **together**:

```text
Question: How do I reset my password?
Document: To reset your password, open Settings...
              ↓
        Reranker model
              ↓
        Relevance score: 0.94
```

The score is used to rank candidates. **It is not necessarily a probability.**

### 13.3 Embedding model vs reranker

| Feature | Embedding model | Reranker |
|---|---|---|
| Input | Text separately | Query + document together |
| Output | Vector | Relevance score |
| Main purpose | Find similar documents | Reorder candidates by relevance |
| Typical use | Initial search | Second-stage ranking |
| Cost per candidate | Usually lower | Usually higher |

### 13.4 Example

Question: **"How do I refresh my JWT access token?"**

Initial vector search:

| Candidate | Initial rank |
|---|---:|
| JWT token generation | 1 |
| User registration API | 2 |
| Refresh token endpoint | 3 |
| Password reset API | 4 |

After reranking:

| Candidate | Reranked position |
|---|---:|
| Refresh token endpoint | 1 |
| JWT token generation | 2 |
| User registration API | 3 |
| Password reset API | 4 |

The refresh token endpoint is now first because it directly answers the question.

### 13.5 Key takeaway

> **Search finds candidates. Reranking judges those candidates more carefully. The LLM uses the final selected context to answer.**

A reranker is often a separate model, but it doesn't have to be — some systems use LLMs for reranking too, though specialised reranker models are common for efficiency.

---

## 14. Decision Point — Start Implementing

**Decision:** start implementation now, before Lesson 10. Enough theory had been covered to build a first working pipeline; you don't need to understand every advanced concept before writing code, and building something shows how embeddings, chunking, vector search and retrieval fit together.

### 14.1 First project — Document Q&A

An application where you upload documents and ask questions about their content.

**Tech stack:** Python, an embedding model, PostgreSQL + pgvector, an LLM API, FastAPI once the core pipeline works.

**Implementation roadmap (conceptual steps):**

1. **Document ingestion** — load a `.txt`/`.md` file, split into chunks, store the chunk text
2. **Generate embeddings** — convert each chunk into a vector
3. **Store vectors** — save text, embedding and metadata in PostgreSQL using pgvector
4. **Build retrieval** — convert a user's question into an embedding, retrieve the top 5 chunks
5. **Generate an answer** — pass question + retrieved text to an LLM with instructions to answer using the supplied context
6. **Test the pipeline** — ask both answerable and unanswerable questions; check whether retrieval finds the right chunks and whether the LLM answers accurately

### 14.2 Staged progression

| Stage | What you'll build |
|---|---|
| **V1** | Basic RAG: chunking, embeddings, vector search, LLM |
| **V2** | Metadata filtering and hybrid search |
| **V3** | Reranking and improved retrieval |
| **V4** | Evaluation: Recall@K, Precision@K, faithfulness |
| **V5** | FastAPI endpoints, document management, production improvements |

**Recommendation:** start with V1 using PostgreSQL + pgvector rather than an in-memory vector store, for practical experience on the database side of RAG — directly relevant to a backend background. Then continue with Lesson 10 (evaluation) and use those metrics to improve the implementation.

### 14.3 Implementation notes that emerged along the way

- **Naive paragraph splitting produces bad chunks** — headings end up in separate chunks, and a long run of paragraphs can create a huge chunk. Real chunking combines paragraphs up to a target size with a small overlap.
- **The character counts used for learning are only to teach the mechanism.** In a real RAG system you think in **tokens**, because LLM context windows and embedding models operate around tokenized text. The target shape is `document → paragraph/sentence boundaries → token-aware chunking → ~300–800 tokens → embedding`. The exact size isn't universal.
- **Chunking happens before embedding** — you need `Chunk 1 → Embedding 1` so vector search can identify *which specific part* of the document is relevant.
- **Batch embedding** is worth using later; one API request per chunk is wasteful.
- **The embedding dimension must be fixed in the database.** If a model produces 384-dim vectors, the pgvector column is `VECTOR(384)`; if 1536, then `VECTOR(1536)`.

---

## 15. Stack Decisions — Embedding and Generation Providers

Decisions made while starting implementation.

### 15.1 Groq for generation, but not for embeddings

Groq provides fast LLM inference, but its documented model catalogue does **not** include an embedding model. So Groq is usable for the **generation** stage only.

### 15.2 Option considered — local embeddings

A local `sentence-transformers` model such as `all-MiniLM-L6-v2`:

- Free to run locally, no API cost
- Produces **384-dimensional** vectors (so pgvector would be `VECTOR(384)`)
- Downloads the model on first run, then runs locally
- Trade-off: local compute requirements, model management, deployment complexity

This is a sensible free alternative if you want to avoid any paid embedding API.

### 15.3 Decision — keep OpenAI embeddings

```text
Component            Technology                        Cost
─────────────────────────────────────────────────────────────
Embeddings           OpenAI text-embedding-3-small     $0.02 / 1M input tokens
Vector database      PostgreSQL + pgvector             Free, self-hosted
LLM generation       Groq                              Free tier, usage limits
Backend (later)      FastAPI                           —
```

- `text-embedding-3-small` produces **1,536-dimensional** vectors by default.
- Approximate embedding-only costs: 10,000 tokens ≈ $0.0002; 100,000 tokens ≈ $0.002; 1 million tokens ≈ $0.02. These exclude LLM generation costs.
- OpenAI API access is billed by usage; do not assume free credits.
- Conclusion: embeddings are inexpensive for a small learning project, so keep the original OpenAI-based embedding code and use Groq's free tier for generation.

---

## 16. Roadmap Ahead — Not Yet Covered

The following are planned/planned-but-not-yet-taught.

### 16.1 Lesson 10 — RAG Evaluation (next)

The critical engineering question:

> **How do you know whether your RAG system is actually good?**

Planned coverage: **Recall@K, Precision@K, MRR, context relevance, faithfulness, answer correctness**, and using an LLM to evaluate a RAG pipeline.

Evaluation structure from the roadmap:

```text
             RAG Evaluation
                   │
          ┌────────┴────────┐
          │                 │
     Retrieval          Generation
      quality             quality
          │                 │
     ┌────┴────┐       ┌────┴─────┐
     │         │       │          │
   Recall  Precision Faithfulness Relevance
```

- **Retrieval quality metrics:** Recall@K, Precision@K, MRR, NDCG
- **Generation quality metrics:** context relevance, context recall, answer relevance, faithfulness, groundedness

Evaluation dataset shape:

```text
question
expected_answer
relevant_document
```

> Evaluation is described as **very important** for RAG knowledge that is useful in interviews and production.

### 16.2 Still ahead from the original roadmap

- **Advanced RAG:** query rewriting, multi-query RAG, HyDE (Hypothetical Document Embeddings), parent-child retrieval, contextual retrieval, Graph RAG (moving from `chunks → vectors` toward `entities → relationships → graph`)
- **Production RAG:** caching, async ingestion, Kafka-driven ingestion, background workers, rate limiting, auth, multi-tenancy, observability, cost tracking, streaming, retries, dead-letter queues, document versioning
- **Project phases:** FastAPI service, conversation history, citations
- **Frameworks (deliberately last):** LangChain, LlamaIndex, LangGraph, agents, tools, structured output, function/tool calling

### 16.3 Open question carried forward

> **How do we know your RAG system is good?** — Answering this is the next major step.

---

## 17. Master Mental Models and Reference Tables

### 17.1 The complete RAG architecture learned so far

```text
                        INGESTION
                            │
                        Documents
                            ↓
                         Chunking
                            ↓
                    Embedding model
                            ↓
                 Vector + chunk text
                            ↓
                   Vector database
                            │
                  ───────────┼──────────
                            │
                            ↓
                         QUERY
                            │
                       User question
                            │
              ┌─────────────┴─────────────┐
              ↓                           ↓
       Query embedding               Query terms
              ↓                           ↓
       Vector search                    BM25
              └─────────────┬─────────────┘
                            ↓
                      Result fusion
                            ↓
                 Metadata filtering
                            ↓
                   Top candidates
                            ↓
                      Reranker
                            ↓
                   Best evidence
                            ↓
                   Context builder
                            ↓
                Question + Context
                            ↓
                           LLM
                            ↓
                         Answer
```

### 17.2 The eight mental models

1. **Embedding** — convert text into a vector so we can compare semantic similarity.
2. **Vector database** — store and efficiently retrieve vectors and their associated content.
3. **Chunking** — create useful retrieval units from large documents.
4. **Retrieval** — find the evidence that should be given to the LLM.
5. **Hybrid retrieval** — combine semantic matching with exact-term matching.
6. **Reranking** — take retrieved candidates and judge/reorder them more precisely.
7. **LLM generation** — use the retrieved evidence and question to produce the final answer.
8. **The fundamental principle:**

```text
LLM knowledge
      +
Retrieved external context
      ↓
Grounded answer
```

### 17.3 The core distinction to remember

> **Retrieval finds the evidence. Generation uses the evidence.**

### 17.4 Three responsibilities

```text
Embedding model  →  FIND
Vector database  →  RETRIEVE
LLM              →  GENERATE
```

### 17.5 Two separations never to blur

| Separation | Meaning |
|---|---|
| **Retrieval vs Attention** | Retrieval selects what enters the context (outside the LLM). Attention determines how the model uses relationships within the context (inside the transformer). |
| **Retrieval vs Generation** | Retrieval: what information should the LLM get? Generation: what should the LLM say with it? |

### 17.6 Data purpose reference

| Data | Purpose |
|---|---|
| Chunk text | Give to the LLM |
| Embedding | Find relevant chunks |
| Metadata | Filter / identify chunks |

### 17.7 Scoring fusion reference

| Approach | Basis | Note |
|---|---|---|
| Cosine similarity | Vector geometry | scale depends on model/metric/normalization |
| BM25 | Term statistics | term frequency, IDF, document length |
| RRF | **Rank positions** | preferred over comparing incompatible raw scores |

---

## 18. Common Misconceptions — Correct Answers

| Statement | Verdict | Correction |
|---|---|---|
| "An LLM stores every sentence from its training data and searches those sentences when answering." | **False** | Knowledge/behavior is encoded in model parameters; generation is based on supplied context plus learned parameters. |
| "A token is always one word." | **False** | Tokens can be whole words, subwords, punctuation, whitespace, special tokens. |
| "If I retrieve more documents for RAG, the answer will always become better." | **False** | More context adds noise, raises cost/latency, and can make relevant info harder to use (lost in the middle). |
| "1 word = 1 token." | **False** | Rough intuition only: ~1 token ≈ ¾ English word; varies hugely by content. Measure with the real tokenizer. |
| "The embedding contains the original text / is a dictionary." | **False** | It is a distributed representation; you must store the chunk text separately. |
| "Bigger embedding dimension is automatically better." | **False** | Quality depends on model and training, not dimension count. |
| "You can compare vectors from different embedding models." | **False** | They live in different spaces; switching models requires re-embedding and re-indexing. |
| "Similarity 0.94 means 94% correct." | **False** | A similarity score is not a probability; thresholds must be tuned empirically. |
| "Dimension 1 = password, dimension 2 = payment." | **False** | Information is distributed across all dimensions. |
| "Attention retrieves the relevant document." | **False** | Attention is inside the transformer; retrieval happens outside the LLM. |
| "A transformer is just attention." | **False** | A block also has Add & Normalize, a Feed-Forward Network, residual connections, etc., stacked N times. |
| "Token IDs and embeddings are the same." | **False** | Token IDs are vocabulary identifiers; embeddings are numerical representations. |
| "Context = knowledge." | **False** | Putting something in context does not make the model permanently learn it. |
| "RAG eliminates hallucinations." | **False** | If retrieval returns the wrong document, you get a confidently wrong answer. |
| "RAG = LLM + Vector DB." | **Beginner definition** | Better: retrieval system + context construction + LLM generation. |
| "The LLM searches the vector database itself." | **False** | The application orchestrates retrieval and supplies the context. |
| "The reranker is another full search." | **False** | It re-evaluates and reorders the candidate set produced by initial retrieval. |
| "Add a 500-token chunk because that's standard." | **False** | A good chunk is a self-contained unit of information; token size is a constraint, meaning is the goal. |
| "Top-K = 5 is always right." | **False** | Tune and evaluate K; too few misses information, too many adds noise. |
| "Vector search will always return the answer." | **False** | Design so that "no relevant evidence found" is a valid outcome. |

---

## 19. Self-Check Question Bank

### After Lesson 1 (LLM fundamentals)

1. What does an LLM fundamentally do, and what does it *not* do?
2. Why does an LLM "know" things if it doesn't search a database?
3. What is a parameter, and how are parameters learned?
4. What is the context window made of?
5. Why is "context ≠ knowledge" important for RAG?
6. Under what conditions does an LLM hallucinate, and why doesn't RAG remove that risk?

### After Lesson 2 (Tokens)

1. What is a token, and what can it represent?
2. Why do tokenizers use subwords instead of whole words?
3. Distinguish token ID from embedding.
4. Why does token count determine the shape of a RAG system?
5. What is a token budget, and why must you reserve space for output?
6. Explain "lost in the middle".
7. Why is tokenisation of source code, SQL, JSON and YAML different in character from prose?
8. Why must you count tokens with the actual model's tokenizer?

### After Lesson 3 (Transformers)

1. What problem does attention solve that a lookup table cannot?
2. What are Query, Key and Value?
3. What is self-attention, and why "self"?
4. Why does attention not retrieve documents?
5. What are multi-head attention heads for?
6. What else lives inside a transformer block?
7. Why do transformers need positional information?

### After the FFN detour

1. What is the role of the feed-forward network relative to attention?
2. Why is the FFN applied per token?

### After Lesson 4 (Embeddings)

1. Why would "How can I reset my password?" and "I forgot my credentials. How do I regain access?" have similar embeddings?
2. What is the difference between token ID and embedding?
3. Why don't we store only the embedding and discard the original chunk?
4. What is the difference between vector similarity and LLM attention?
5. Why does cosine similarity work for comparing embeddings?
6. Why does similarity not equal relevance?

### After Lesson 5 (Vector search)

1. What is vector search, and what is Top-K?
2. Why prefer vector search over keyword-only search?
3. What is a vector database, and what does pgvector add to PostgreSQL?
4. Why are HNSW/ANN needed at scale?
5. What does the vector DB return, and what does the LLM receive?

### After Lesson 6 (Pipeline)

1. Why do we create an embedding for the user's question?
2. Does the LLM receive the embedding or the original retrieved text?
3. Why store both `content` and `embedding`?
4. Why not compare the query against every vector with 1M chunks?
5. What are the two separate RAG pipelines?
6. Who orchestrates retrieval — the LLM or the application?

### After Lesson 7 (Chunking)

1. Why can a 10,000-token chunk be bad even if the LLM can fit it in its context window?
2. What problem does chunk overlap solve?
3. Why might source code need a different chunking strategy from a normal PDF?
4. Why do we chunk *before* creating embeddings?
5. A query retrieves a tiny child chunk lacking enough context — what advanced technique helps?
6. What makes a "good chunk"?

### After Lesson 8 (Embedding models)

1. What is the difference between an embedding model and a chat LLM?
2. When is a larger embedding dimension actually a problem?
3. What breaks if you embed documents with model A and queries with model B?
4. What is required operationally to change embedding models in production?
5. API vs local embeddings — trade-offs?
6. Why is a similarity score not a probability?
7. How would you choose between two embedding models?
8. What is incremental indexing, and why does it matter?

### After Lesson 9 (Retrieval)

1. Why isn't `Top-K = 5` always correct?
2. What's the difference between a similarity threshold and Top-K?
3. Why is metadata filtering essential in multi-tenant RAG?
4. What kind of query might BM25 handle better than semantic search?
5. Why rerank after initial retrieval instead of reranking millions of chunks?
6. Why does it matter whether a wrong answer came from retrieval or generation?
7. Why can't you just add a cosine score to a BM25 score?
8. What problem does RRF solve?

### After the reranker lesson

1. Is a reranker another search, or a model? What is it usually?
2. How does a cross-encoder differ from a bi-encoder/embedding model?
3. Why is reranking applied to a small candidate set?

---

_End of notes._
