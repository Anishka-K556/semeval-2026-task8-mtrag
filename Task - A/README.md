# Task A — Conversational Passage Retrieval Pipeline

**SemEval 2026 Task 8: Multi turn RAG Subtask A — Stage 1 (Query Rewriting) + Stage 2 (Hybrid Retrieval + Reranking)**

---

## Pipeline overview

```
Multi-turn conversation
    |
    v (Stage 1)
[Gemma-2-2B-IT few-shot rewriting]
    |
    v (Stage 2)
Standalone rewritten query
    |
    +-------> BM25           top-150  |
    +-------> BGE-base-en    top-150  +--> RRF (k=60) --> top-100
    +-------> E5-base-v2     top-150  |
    |
    v
Ensemble cross-encoder reranking:
    BGE-reranker-large  x 0.40
    mxbai-rerank-large  x 0.35
    ms-marco-MiniLM-L12 x 0.25
    |
    v
Top-10 passages with scores
```

Weights were tuned on the dev set using nDCG@5.

---

## Repo structure

```
Task-A/
├── config.json                      <- all hyperparameters and paths
├── query_rewriting/
│   └── rewrite.py                   <- Stage 1: Gemma-2-2B-IT query rewriting
├── preprocessing/
│   └── generate_embeddings.py       <- one-time corpus encoding (BGE + E5)
├── retrieval/
│   └── pipeline.py                  <- Stage 2: full Task A pipeline
├── utils/
│   └── config_loader.py             <- shared config loading helper
├── evaluate.py                      <- nDCG@5/10, Recall@5/10 evaluation
├── combine_results.py               <- merge per-domain files into submission
└── README.md
```

---

## Setup

```bash
pip install sentence-transformers rank-bm25 faiss-cpu torch tqdm transformers accelerate bitsandbytes
```

For GPU (recommended — required for Gemma):
```bash
pip install faiss-gpu
```

---

## Configuration

Edit **only** the `base_path` in `config.json`:

```json
{
  "base_path": "/path/to/SemEval",
  ...
}
```

All other paths are derived automatically. Expected data layout:

```
SemEval/
├── corpus/
│   ├── clapnq.jsonl
│   ├── cloud.jsonl
│   ├── fiqa.jsonl
│   └── govt.jsonl
├── queries/
│   ├── clapnq_lastturn.jsonl    <- raw last-turn queries (input to Stage 1)
│   ├── cloud_lastturn.jsonl
│   ├── fiqa_lastturn.jsonl
│   └── govt_lastturn.jsonl
│   (after Stage 1:)
│   ├── clapnq_rewrite.jsonl     <- rewritten queries (input to Stage 2)
│   ├── cloud_rewrite.jsonl
│   ├── fiqa_rewrite.jsonl
│   └── govt_rewrite.jsonl
├── qrels/
│   ├── clapnq/dev.tsv
│   ├── cloud/dev.tsv
│   ├── fiqa/dev.tsv
│   └── govt/dev.tsv
└── Embeddings/                  <- created by generate_embeddings.py
    ├── clapnq_bge_base.pkl
    ├── clapnq_e5_base.pkl
    └── ...
```

---

## Step-by-step usage

### Step 0 — Generate corpus embeddings (once per domain)

```bash
python preprocessing/generate_embeddings.py --domain clapnq
python preprocessing/generate_embeddings.py --domain ibmcloud
python preprocessing/generate_embeddings.py --domain fiqa
python preprocessing/generate_embeddings.py --domain govt
```

Use `--skip-existing` to skip domains whose `.pkl` files already exist.

### Step 1 — Rewrite queries with Gemma

```bash
# Single domain
python query_rewriting/rewrite.py --domain clapnq

# All domains at once
python query_rewriting/rewrite.py --domain all
```

This reads `<domain>_lastturn.jsonl` and writes `<domain>_rewrite.jsonl`
in the same queries directory. Requires a GPU with ~3 GB+ VRAM (uses 4-bit
quantisation by default). On CPU it will work but is very slow.

### Step 2 — Run the retrieval pipeline (dev evaluation)

```bash
# Single domain
python retrieval/pipeline.py --domain clapnq

# All domains (shares models, much faster)
python retrieval/pipeline.py --domain all
```

Each domain writes `results/<domain>_taskA_results.jsonl` and prints
Recall@5, nDCG@5, Recall@10, nDCG@10 at the end.

### Step 2 (alternative) — Run on test queries for submission

```bash
# If test queries are in the same format as dev queries:
python retrieval/pipeline.py --domain clapnq --test \
    --query-path /path/to/clapnq_test_queries.jsonl

# Or if test queries are already in the default query_paths location:
python retrieval/pipeline.py --domain all --test
```

`--test` skips evaluation (no qrels needed) and writes the same JSONL format
output ready for submission.

### Step 3 — Combine into submission file

```bash
python combine_results.py
```

Merges the four per-domain result files into `final_submission_taskA.jsonl`.

### Step 4 — Evaluate (dev set)

```bash
# Single domain
python evaluate.py \
    --results results/clapnq_taskA_results.jsonl \
    --qrels   /path/to/qrels/clapnq/dev.tsv

# All domains + macro average
python evaluate.py --submission final_submission_taskA.jsonl
```

---

## Key hyperparameters

| Parameter | Value | Location in config |
|---|---|---|
| RRF k | 60 | `retrieval.rrf_k` |
| Candidates per retriever | 150 | `retrieval.top_retrieval` |
| After RRF fusion | 100 | `retrieval.top_fusion` |
| Final top-K | 10 | `retrieval.final_top_k` |
| BGE-reranker-large weight | 0.40 | `reranking.weights` |
| mxbai-rerank-large weight | 0.35 | `reranking.weights` |
| ms-marco-MiniLM weight | 0.25 | `reranking.weights` |

---

## Dev set results (rewritten queries, dev qrels)

| Domain | Recall@5 | nDCG@5 |
|--------|----------|--------|
| ClapNQ | 0.6875 | 0.4125 |
| IBM Cloud | 0.5266 | 0.3260 |
| FiQA | 0.5389 | 0.2921 |
| Govt | 0.6219 | 0.3938 |

---

## Submission output format (one line per query)

```json
{
  "task_id": "q123",
  "Collection": "clapnq",
  "contexts": [
    {"document_id": "doc_456", "score": 0.91},
    ...
  ]
}
```
