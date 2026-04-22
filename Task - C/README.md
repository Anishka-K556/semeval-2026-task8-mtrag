# Task-C — Answer Generation Pipeline

Generates answers for SemEval 2026 RAG Design Challenge (Task C) using a FLAN-T5 model grounded on pre-retrieved passages from Task A.

---

## Directory Structure

```
Task-C/
├── config.json                  ← All configurable parameters
├── run.py                       ← Top-level entry point
├── evaluate.py                  ← ROUGE-L metric computation + sample display
├── utils.py                     ← Shared helpers (load_config)
├── generation/
│   ├── __init__.py
│   └── generate.py              ← Model loading, prompt building, inference, output saving
└── postprocessing/
    ├── __init__.py
    └── normalize.py             ← Output cleaning + evaluation string normalization
```

---

## Prerequisites

Install dependencies before running:

```bash
pip install transformers==4.41.0 torch rouge-score==0.1.2 sentencepiece==0.2.0 accelerate==0.30.0
```

A CUDA-capable GPU is strongly recommended. The pipeline falls back to CPU but will be significantly slower (~15 min per query vs ~0.5 sec on GPU).

---

## Connecting Task A Output to Task C

Task C expects a JSONL file of query tasks produced by Task A. Each record must contain a `contexts` list of pre-retrieved passages with `document_id` and `score` fields.

To point Task C at your Task A output, open `config.json` and update **`paths.query_task_path`**:

```json
"paths": {
    "query_task_path": "/path/to/your/task_a_output.jsonl",
    ...
}
```

For example, if Task A wrote its results to `/content/Kairo_taskA.jsonl`, the default value already reflects that. Change it to match wherever your Task A output is saved.

While you have `config.json` open, also verify the other paths match your environment:

| Key | Description |
|---|---|
| `paths.query_task_path` | Task A output — queries + retrieved passage IDs |
| `paths.corpus_paths` | Raw corpus JSONL files (clapnq, cloud, fiqa, govt) |
| `paths.output_path` | Where Task C predictions will be written |

---

## Configuration Reference

All tunable parameters live in `config.json`. Common values you may want to change:

```json
{
  "model_name": "google/flan-t5-large",
  "top_k_context": 3,
  "token_limits": {
    "max_input_tokens": 1024,
    "max_new_tokens": 60
  },
  "generation": {
    "num_beams": 3,
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,
    "early_stopping": true
  },
  "paths": {
    "query_task_path": "/content/Kairo_taskA.jsonl",
    "output_path": "/content/final_results.jsonl",
    "corpus_paths": [
      "/content/clapnq.jsonl",
      "/content/cloud.jsonl",
      "/content/fiqa.jsonl",
      "/content/govt.jsonl"
    ]
  }
}
```

To use a larger model (requires 16 GB+ GPU), change `model_name` to `"google/flan-t5-xl"`.

---

## Order of Execution

Run the files in this order:

### Step 1 — Generation

Loads the model, builds grounded prompts, runs inference, and writes predictions to `paths.output_path`.

```bash
python generation/generate.py
```

### Step 2 — Evaluation

Reads the predictions file and computes mean ROUGE-L F1. Only records with a `reference_answer` field are scored (dev/validation splits). Test-set runs will report `"No valid reference answers available."` — this is expected.

```bash
python evaluate.py
```

### Run Both Steps Together

Use the top-level entry point to run generation followed by evaluation in one command:

```bash
python run.py
```

To skip generation and evaluate an existing predictions file:

```bash
python run.py --eval-only
```

To use a custom config file:

```bash
python run.py --config path/to/my_config.json
```

---

## Output Format

Each line in the output JSONL file contains:

```json
{
  "task_id": "...",
  "Collection": "...",
  "original_query": "...",
  "rewritten_query": "...",
  "contexts": [
    { "document_id": "...", "text": "...", "score": 0.95 }
  ],
  "predictions": [
    { "text": "Generated answer here" }
  ]
}
```

Unanswerable queries are normalised to `"I don't know"` regardless of the model's exact phrasing.
