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

## Key Parameters

| Parameter | Value | Config key |
|---|---|---|
| Generation model | google/flan-t5-large | `model_name` |
| Top-K context passages | 3 | `top_k_context` |
| Max input tokens | 1024 | `token_limits.max_input_tokens` |
| Max new tokens | 60 | `token_limits.max_new_tokens` |
| Beam search width | 3 | `generation.num_beams` |
| No-repeat n-gram size | 3 | `generation.no_repeat_ngram_size` |
| Length penalty | 1.0 | `generation.length_penalty` |
| Early stopping | true | `generation.early_stopping` |

---

## Dev Set Results

Metrics are computed per domain on the dev split. **RB_agg** is the aggregated ROUGE-BERT score, **RL_F** is ROUGE-L F1, and **RB_llm** is the LLM-based ROUGE-BERT variant. Avg. Length is the mean character length of generated answers.

| Domain    | N   | RB_agg | RL_F   | RB_llm | Avg. Length (chars) |
|-----------|-----|--------|--------|--------|---------------------|
| IBM Cloud | 131 | 0.2663 | 0.5588 | 0.2905 | 55.8                |
| FiQA      | 77  | 0.2187 | 0.6645 | 0.2500 | 64.8                |
| ClapNQ    | 142 | 0.1893 | 0.4437 | 0.2377 | 45.7                |
| Govt      | 157 | 0.1734 | 0.4554 | 0.1803 | 86.6                |

**Overall:** Harmonic mean of **0.3198** (RL_F = 0.6011), ranking **26th out of 29 teams**.

Analysis from the paper notes that strong factual grounding through anchor prompting contributed to the RL_F score, while lower semantic similarity scores reflect limitations of FLAN-T5-large in capturing nuanced responses. Retrieval noise and over-conservative abstention (excessive "I don't know" responses) were the primary sources of error.

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
