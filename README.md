# TechSSN at SemEval-2026 Task 8: MTRAG

**Retrieval and Generation using Ensemble Re-encoders and Anchor Prompting**

> Sri Sivasubramaniya Nadar College of Engineering, Chennai, India

> Anishka K · Anne Jacika J · Guruprakash K · Rajalakshmi Sivanaiah · S. Angel Deborah

---

## About

This repository contains the full system submitted to [SemEval-2026 Task 8: MTRAG-UN](https://semeval.github.io/SemEval2026/tasks.html) — a shared task on multi-turn conversational retrieval-augmented generation. The benchmark consists of 666 multi-turn conversational tasks across six domains, targeting unanswerable, underspecified, non-standalone, and unclear queries.

We participated in two subtasks:

| Subtask | Description | Status |
|---------|-------------|--------|
| **Task A** — Document Retrieval | Retrieve the top-10 relevant passages for each conversational query | Complete |
| **Task C** — Full RAG Pipeline | Generate a grounded, faithful response using retrieved passages | In progress |

The system covers four domains: **ClapNQ**, **IBM Cloud**, **FiQA**, and **Govt**.

---

## System overview

```
Multi-turn conversation
        |
        v  [Task A — Stage 1]
Gemma-2-2B-IT
(4-bit NF4 quantisation · few-shot · greedy decoding)
        |
        v
Standalone rewritten query
        |
        +-----------> BM25 (lexical)                    top-150  |
        +-----------> BAAI/bge-base-en-v1.5  (dense)    top-150  +--> RRF (k=60) --> top-100
        +-----------> intfloat/e5-base-v2    (dense)    top-150  |
        |
        v  [Task A — Stage 2]
Ensemble cross-encoder reranking
        BAAI/bge-reranker-large              x 0.40
        mixedbread-ai/mxbai-rerank-large-v1  x 0.35
        cross-encoder/ms-marco-MiniLM-L-12   x 0.25
        |
        v
Top-10 passages                             [Task A output]
        |
        v  [Task C]
google/flan-t5-large
(anchor prompting · top-3 passages · FP16)
        |
        v
Post-processing + IDK normalization
        |
        v
Grounded response                           [Task C output]
```

---

## Repository structure

```
SemEval-2026-Task8-MTRAG/
│
├── README.md                            <- this file
├── config.json                          <- all paths and hyperparameters (shared)
│
├── utils/
│   └── config_loader.py                 <- shared config loading helper
│
├── Task-A/
│   ├── README.md                        <- Task A setup and usage
│   ├── query_rewriting/
│   │   └── rewrite.py                   <- Stage 1: Gemma query rewriting
│   ├── preprocessing/
│   │   └── generate_embeddings.py       <- one-time corpus encoding (BGE + E5)
│   ├── retrieval/
│   │   └── pipeline.py                  <- Stage 2: hybrid retrieval + reranking
│   ├── evaluate.py                      <- Recall@5/10, nDCG@5/10 evaluation
│   └── combine_results.py               <- merge per-domain files into submission
│
├── Task-C/
│   ├── README.md                        <- Task C setup and usage (in progress)
│   ├── generation/
│   │   └── generate.py                  <- FLAN-T5-Large anchor prompting
│   ├── postprocessing/
│   │   └── normalize.py                 <- IDK normalization
│   └── evaluate.py                      <- RBllm, RBagg, RLF, IDK metrics
│
└── paper/
    └── TechSSN_SemEval2026_Task8.pdf
```

`config.json` and `utils/` sit at the root and are shared by both tasks. Neither task folder is self-contained — both read config from the root.

---

## Setup

```bash
pip install sentence-transformers rank-bm25 faiss-cpu torch tqdm transformers accelerate bitsandbytes
```

For GPU (recommended — required for Gemma rewriting):

```bash
pip install faiss-gpu
```

---

## Configuration

Edit only `base_path` in `config.json`:

```json
{
  "base_path": "/path/to/your/SemEval/data",
  ...
}
```

All corpus, query, qrel, embedding, and output paths resolve automatically. See the individual task READMEs for the expected data layout.

---

## Quick start

Full instructions for each task are in their respective READMEs. The high-level run order is:

```bash
# 1. Generate corpus embeddings (once)
python Task-A/preprocessing/generate_embeddings.py --domain all

# 2. Rewrite conversational queries
python Task-A/query_rewriting/rewrite.py --domain all

# 3. Run hybrid retrieval + reranking
python Task-A/retrieval/pipeline.py --domain all

# 4. Combine into submission file
python Task-A/combine_results.py
```


## Reference Paper

System description paper is available in [`paper/`](paper/TechSSN_SemEval2026_Task8.pdf).

```bibtex
@inproceedings{techssn-semeval2026-task8,
  title     = {TechSSN at SemEval-2026 Task 8: MTRAG Retrieval and Generation
               using Ensemble Re-encoders and Anchor Prompting},
  author    = {Anishka K and Anne Jacika J and Guruprakash K and
               Rajalakshmi Sivanaiah and S. Angel Deborah},
  booktitle = {to be updated},
  year      = {2026}
}
```
