# AetherV
A version-aware, incrementally maintained vector database that treats knowledge as a continuously evolving graph rather than a static collection of embeddings.
# Project AetherV Evolution Engine (AEE)

## Vision

Build the first retrieval-aware, version-aware, incrementally maintained vector database capable of operating on continuously changing knowledge without full reindexing.

Current vector databases optimize:

* Similarity search
* ANN indexing
* Storage efficiency

AetherV Evolution Engine optimizes:

* Knowledge freshness
* Incremental updates
* Retrieval correctness under change
* Autonomous index maintenance

---

# Core Thesis

The future bottleneck of RAG is not retrieval speed.

The bottleneck is:

"How can a retrieval system stay correct while its knowledge continuously changes?"

Current systems:

Document changes
→ Rechunk
→ Re-embed
→ Reindex

AetherV:

Document changes
→ Semantic diff
→ Impact prediction
→ Localized updates
→ Retrieval remains correct

---

# Product Definition

Category:

Version-Aware Dynamic Retrieval Engine

Tagline:

Git for Knowledge + Vector Database

Primary Users:

* Enterprise RAG
* Documentation systems
* Agentic systems
* Knowledge management platforms
* Real-time data platforms

---

# Key Differentiators

## 1. Semantic Change Engine

Determine what actually changed.

Input:

Document v1
Document v2

Output:

* Added concepts
* Modified concepts
* Deleted concepts
* Dependency impact

Goal:

Avoid unnecessary embedding generation.

---

## 2. Retrieval Impact Predictor

Novel research component.

Question:

Will this change affect retrieval?

Example:

"128GB RAM"
→
"129GB RAM"

Embedding changes.

Retrieval behavior likely does not.

Decision:

Skip expensive update.

Expected savings:

70-95% embedding reduction.

---

## 3. Version-Aware Retrieval

Every chunk becomes temporal.

Chunk schema:

{
chunk_id,
version_id,
valid_from,
valid_to,
parent_version
}

Supports:

* Historical retrieval
* Change tracking
* Temporal QA

---

## 4. Dependency Graph

Knowledge becomes a graph.

Chunk
→ Summary
→ RAPTOR node
→ KG entity
→ Agent memory

If node changes:

Automatically identify stale descendants.

---

## 5. LSM Vector Index

Inspired by LSM-VEC.

Structure:

L0 = recent updates

L1 = warm data

L2 = stable data

L3 = archive

Advantages:

* Fast inserts
* Fast deletes
* No global rebuilds

---

# Research Goals

Goal 1

Reduce embedding regeneration by 90%.

Goal 2

Reduce index rebuild operations to near zero.

Goal 3

Maintain retrieval accuracy >99% of full reindex baseline.

Goal 4

Support continuous ingestion at enterprise scale.

Goal 5

Achieve sub-second update propagation.

---

# System Architecture

Layer 0
Source Connectors

Layer 1
CDC Engine

Layer 2
Semantic Diff Engine

Layer 3
Retrieval Impact Predictor

Layer 4
Embedding Manager

Layer 5
Version Store

Layer 6
Dependency Graph

Layer 7
LSM Vector Index

Layer 8
Retrieval API

---

# Work Packages

WP-1 Foundation

Duration:
2 weeks

Deliverables:

* Monorepo
* CI/CD
* Benchmark framework
* Dataset registry

Success Criteria:

Repeatable experiments.

---

WP-2 Change Detection Engine

Duration:
3 weeks

Tasks:

* File CDC
* Database CDC
* Event ingestion
* Hash-based diffing

Output:

Changed chunk list

Success Criteria:

Detect changes with >99.9% precision.

---

WP-3 Semantic Diff Engine

Duration:
4 weeks

Tasks:

* AST extraction
* Chunk fingerprinting
* Concept extraction
* Semantic similarity graph

Output:

Semantic delta object

Success Criteria:

Correctly classify additions, deletions, modifications.

---

WP-4 Retrieval Impact Predictor

Duration:
6 weeks

Research Track

Tasks:

* Build retrieval benchmark
* Learn retrieval sensitivity
* Predict update necessity

Output:

Impact score

0.0 → no update

1.0 → must update

Success Criteria:

Skip >70% updates while preserving retrieval quality.

---

WP-5 Versioned Storage

Duration:
3 weeks

Tasks:

* Chunk versioning
* Temporal metadata
* Lineage tracking

Success Criteria:

Historical reconstruction support.

---

WP-6 Dependency Graph

Duration:
4 weeks

Tasks:

* Graph schema
* Edge inference
* Incremental propagation

Success Criteria:

Detect all downstream stale nodes.

---

WP-7 Incremental Embedding Engine

Duration:
6 weeks

Tasks:

* Selective re-embedding
* Delta embedding experiments
* Embedding cache

Success Criteria:

90% reduction in embedding workload.

---

WP-8 LSM Vector Index

Duration:
8 weeks

Tasks:

* L0-L3 architecture
* Incremental HNSW
* Compaction engine

Success Criteria:

No full index rebuilds.

---

WP-9 Retrieval Layer

Duration:
4 weeks

Tasks:

* Hybrid search
* Temporal search
* Version-aware ranking

Success Criteria:

Beat baseline RAG retrieval.

---

WP-10 Research Publication

Duration:
Ongoing

Targets:

* arXiv
* VLDB
* SIGIR
* NeurIPS Datasets & Benchmarks

Potential Paper Titles:

Retrieval-Aware Incremental Embedding

Version-Aware Dynamic Vector Retrieval

AetherV: A Knowledge Evolution Engine for Continually Updated RAG Systems

---

# Success Metrics

Embedding Cost Reduction:
Target >90%

Update Latency:
Target <1 second

Index Rebuild Frequency:
Target zero

Retrieval Accuracy Loss:
Target <1%

Storage Overhead:
Target <20%

---

# MVP Scope

MVP includes:

✓ CDC

✓ Semantic diffing

✓ Version tracking

✓ Selective re-embedding

✓ Incremental HNSW

✓ Evaluation suite

MVP excludes:

✗ Multi-node clustering

✗ GPU acceleration

✗ Agent orchestration

✗ RAPTOR integration

✗ Knowledge graph generation

These become Phase 2.

---

# Phase 2

* Multi-node distributed engine
* GPU kernels
* RAPTOR hierarchy
* GraphRAG integration
* Agent memory support
* Real-time streaming ingestion
* Learned ANN routing

---
## Project Charter: Project AetherV (Heterogeneous Vector Engine)

This document establishes the scope, architectural design, and implementation roadmap for building a next-generation, heterogeneous vector database from scratch. Project AetherV shifts away from monolithic database paradigms by separating dynamic control logic from parallelized mathematical acceleration.

---

## 1. Executive Summary: Why, How, and What It Solves

### The "Why" (The Rationale)

Current vector databases face a sharp trade-off: they are either optimized for fast, static similarity searches on hardware accelerators (like GPU/TPU indices) or built for dynamic text processing, metadata filtering, and graph routing on standard CPUs. When a production multi-agent system runs complex pipelines—such as RAPTOR tree traversals, hybrid lexical/dense search, and real-time self-correcting routing loops—monolithic databases create severe latency overhead. They waste valuable GPU compute on sequential pointer-chasing logic or bottleneck the CPU with massive array operations.

### What It Solves (The Core Bottlenecks)

* **The Inter-Node Latency Tax:** Eliminates the serialization and network overhead of bouncing data between standalone graph stores, BM25 engines, and vector indices.
* **The Abstraction Tax:** Prevents framework state bloat (inherent in heavy orchestrators) by managing the agentic state machine directly inside the database control layer.
* **The Accelerator Compiling Problem:** Solves JAX’s rigid requirement for static array shapes during execution by implementing a zero-copy memory bridge over fixed-capacity pre-allocated CPU layouts.

### The "How" (The Architectural Split)

AetherV splits the database into two highly specialized planes operating over shared memory:

```
[ Ingestion / Query Client ]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│               CPU CONTROL PLANE (Polars)               │
│  - HNSW Graph Traversal    - BM25 Token Dictionaries   │
│  - RAPTOR Tree Clustering  - Deterministic Statechart  │
└────────────────────────────┬───────────────────────────┘
                             │  Zero-Copy Apache Arrow Bridge
                             ▼
┌────────────────────────────────────────────────────────┐
│               GPU EXECUTION PLANE (JAX)                │
│  - Fused Cosine Similarity - Batched Cross-Encoder Rerank│
│  - Conformal Masking        - Vectorized Index Scans    │
└────────────────────────────────────────────────────────┘

```

---

## 2. Technical Stack Selection

To achieve maximum throughout and mechanical sympathy with the underlying hardware, the core stack is strictly constrained to high-performance, non-bloated libraries:

* **Control Plane & Memory Layout:** **Polars / Apache Arrow**. Polars provides fast columnar structures on the CPU, and Apache Arrow ensures that data can be exposed to accelerators via raw memory pointers without slow serialization passes.
* **Execution Plane:** **JAX (XLA)**. JAX compiles distance functions, cross-encoder scoring, and conformal mask thresholds into fused, branch-free GPU kernels.
* **Concurrency Model:** **Python `asyncio**`. Manages concurrent query streams and asynchronous actor worker tasks without thread blocking.

---

## 3. Structural Database Schemas

### CPU Memory Map (Polars Schema)

The main collection matrix is tracked in a centralized, contiguous memory table. Every document chunk maps directly to a fixed implicit integer row index ($i$).

| Column Name | Type | Purpose / Description |
| --- | --- | --- |
| `id` | `pl.UInt32` | Global unique identifier / contiguous matrix offset |
| `parent_id` | `pl.Int32` | RAPTOR tree pointer (Points to parent summary ID; `-1` if root) |
| `layer_level` | `pl.UInt8` | RAPTOR level ($0$ = base chunk, $1$ = summary child, etc.) |
| `hnsw_edges` | `pl.List(pl.Int32)` | Padded fixed-length array of neighbor node indices |
| `sparse_tokens` | `pl.List(pl.UInt32)` | Hashed vocabulary bin tokens for BM25 matching |
| `sparse_weights` | `pl.List(pl.Float32)` | Corresponding pre-computed term frequencies |
| `doc_string` | `pl.String` | Raw text block (Retained strictly for synthesis output) |

### GPU Tensor Allocation (JAX Fixed-Capacity Arrays)

The execution plane registers a static block of memory on device initialization to prevent costly re-allocation delays during ingestion.

* `DENSE_MATRIX`: `jax.Array` of shape `[MAX_CAPACITY, FEATURE_DIM]` (`float32`)
* `SPARSE_MATRIX`: `jax.Array` of shape `[MAX_CAPACITY, MAX_TOKEN_PADDINGS]` (`float32`)

---

## 4. The Phased Implementation Roadmap

### Phase 1: The Zero-Copy Memory Bridge (Weeks 1–2)

* **Goal:** Establish zero-overhead data handoffs between CPU records and compiled GPU math.
* **Deliverables:**
* Configure a fixed-capacity Apache Arrow allocation layer holding raw vector representations.
* Implement the core JAX execution kernel for batched Cosine Similarity using `jax.jit` and `jax.vmap`.
* Build the interface layer that reads memory pointers from a Polars chunk partition and maps them into JAX device arrays without copying data strings.



### Phase 2: Dual-Engine Hybrid Search (Weeks 3–4)

* **Goal:** Combine exact dense vector lookups with sparse BM25 mechanics on the accelerator.
* **Deliverables:**
* Create a tokenization helper that hashes text blocks into a uniform, padded sparse array layout.
* Write the fused JAX mathematical kernel that simultaneously computes dense cosine distance and sparse dot-product scores.
* Implement a Reciprocal Rank Fusion (RRF) step directly inside the XLA compilation block to return unified top-K index positions.



### Phase 3: Hierarchical Structures & Context Filtering (Weeks 5–6)

* **Goal:** Add support for structural document grouping (RAPTOR) and statistical compression.
* **Deliverables:**
* Build the offline ingestion pipeline that clusters base text embeddings via a lightweight Gaussian Mixture Model (GMM) and inserts parent summaries back into the Polars table layout.
* Write the CPU traversal router to fetch parent summary references when deep context is requested.
* Implement a JAX-backed Conformal Filtering layer to dynamically threshold and mask out noise based on the scoring variance across retrieved arrays.



### Phase 4: Statechart Orchestration & Agentic RAG (Weeks 7–8)

* **Goal:** Embed the multi-agent routing loop directly into the database control plane.
* **Deliverables:**
* Implement an asynchronous `StatechartSupervisor` that transitions search workflows based on strict routing matrices.
* Build independent actor workers (`VectorSearchActor`, `CriticActor`, `SynthesizerActor`) operating over non-blocking message loops.
* Conduct end-to-end performance tracing to measure query execution times under parallel load constraints.



---

## 5. Verification & Success Criteria

To transition the project from alpha implementation to a production-ready target, the engine must satisfy three objective performance baselines:

1. **Zero Dynamic Re-compilation:** After the initial database setup pass, adding new records or executing search pipelines must result in zero XLA re-compilation events.
2. **Deterministic Graph Routing:** The statechart control plane must demonstrate a $100\%$ validation rate against invalid state transition attempts during loop cycles.
3. **Throughput Scaling:** The combined engine must achieve lower latency scales compared to a traditional monolithic database paired with a separate client-side agent framework when running recursive or self-correcting RAG workflows.

Which specific phase of the roadmap should be detailed first to begin drafting the underlying codebase?

# Ultimate Goal

Transform vector databases from static embedding stores into continuously evolving knowledge systems.

The core artifact is not the vector index.

The core artifact is a continuously maintained knowledge evolution graph.

