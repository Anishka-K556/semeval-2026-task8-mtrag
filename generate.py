"""
generation/generate.py

Responsibilities:
  - Load model (FLAN-T5 or equivalent)
  - Build prompts
  - Run inference
  - Save predictions to JSONL
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from postprocessing.normalize import normalize_output
from utils import load_task_c_config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: dict):
    """
    Load query tasks and merge all corpus documents into a single lookup dict.

    Returns:
        tasks      (list[dict])  — query tasks; each has task_id, queries,
                                   and a ranked `contexts` list.
        corpus_map (dict)        — maps document_id -> raw passage text.
    """
    query_task_path = config["paths"]["query_task_path"]
    corpus_paths    = config["paths"]["corpus_paths"]

    # --- Load query tasks ---
    try:
        with open(query_task_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(tasks)} query tasks from {os.path.basename(query_task_path)}")
    except Exception as e:
        print(f"✗ Failed to load query tasks: {e}")
        tasks = []

    # --- Load and merge corpus documents ---
    corpus_map = {}
    for p in corpus_paths:
        try:
            with open(p, "r") as f:
                count = 0
                for line in f:
                    obj = json.loads(line)
                    # Some corpora use 'id', others '_id' — handle both.
                    doc_id = obj.get("id") or obj.get("_id")
                    corpus_map[doc_id] = obj.get("text", "")
                    count += 1
                print(f"✓ Loaded {count} documents from {os.path.basename(p)}")
        except Exception as e:
            print(f"✗ Failed to load {os.path.basename(p)}: {e}")

    return tasks, corpus_map


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_grounded_prompt(task: dict, corpus: dict, config: dict) -> str:
    """
    Build a grounded prompt for a single task.

    Design decisions:
      - Uses `rewritten_query` when available; falls back to `original_query`.
      - Limits context to Top-K passages (from config) to avoid the
        "lost-in-the-middle" effect.
      - Strips URLs and FAQ headers from passages.
      - Appends an anchor phrase so the decoder is primed to stay grounded.

    Args:
        task   (dict): Single task entry from the query task file.
        corpus (dict): document_id -> passage text lookup.
        config (dict): Loaded configuration dict.

    Returns:
        str: Formatted prompt string ready for tokenisation.
    """
    top_k             = config["top_k_context"]
    prompt_template   = config["prompt_template"]

    # Prefer the rewritten query for cleaner semantics.
    query = task.get("rewritten_query") or task.get("original_query", "")

    # Retrieve ranked context list (pre-computed).
    retrieved_contexts = task.get("contexts", [])

    # Build numbered source blocks from the top-K passages.
    passages = []
    for i, ctx in enumerate(retrieved_contexts[:top_k]):
        doc_id = ctx.get("document_id")
        if doc_id in corpus:
            # Strip URL tails and FAQ sections that introduced noise.
            clean_text = corpus[doc_id].split("http")[0].split("FAQs")[0]
            passages.append(f"SOURCE {i+1}: {clean_text.strip()}")

    context = "\n".join(passages) if passages else "No sources available."

    return prompt_template.format(context=context, query=query)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(config: dict):
    """
    Load tokenizer and model onto the appropriate device.

    GPU path: FP16 halves memory footprint and roughly doubles throughput.
    CPU path: default FP32 for numerical stability.

    Returns:
        tokenizer: Loaded AutoTokenizer.
        model:     Loaded AutoModelForSeq2SeqLM in eval mode.
        device:    'cuda' or 'cpu'.
    """
    model_name = config["model_name"]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded with FP16 precision (GPU optimized)")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        print("✓ Model loaded on CPU")

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"\nGPU Memory Status:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Free:      {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")

    return tokenizer, model, device


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(task: dict, corpus: dict, tokenizer, model, device: str, config: dict) -> str:
    """
    Run model inference for a single task and return the raw decoded string.

    Args:
        task      (dict): Single task entry.
        corpus    (dict): document_id -> passage text lookup.
        tokenizer:        HuggingFace tokenizer.
        model:            HuggingFace model.
        device    (str):  'cuda' or 'cpu'.
        config    (dict): Loaded configuration dict.

    Returns:
        str: Raw decoded generation text (before normalization).
    """
    max_input  = config["token_limits"]["max_input_tokens"]
    max_new    = config["token_limits"]["max_new_tokens"]
    gen_cfg    = config["generation"]

    prompt = build_grounded_prompt(task, corpus, config)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_beams=gen_cfg["num_beams"],
            no_repeat_ngram_size=gen_cfg["no_repeat_ngram_size"],
            length_penalty=gen_cfg["length_penalty"],
            early_stopping=gen_cfg["early_stopping"]
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# =============================================================================
# OUTPUT SERIALISATION
# =============================================================================

def build_output_record(task: dict, corpus: dict, gen_text: str, config: dict) -> dict:
    """
    Build a single JSONL output record for a task.

    Attaches the top-K contexts with full passage text for traceability.

    Args:
        task     (dict): Single task entry.
        corpus   (dict): document_id -> passage text lookup.
        gen_text (str):  Normalised prediction text.
        config   (dict): Loaded configuration dict.

    Returns:
        dict: Record ready for json.dumps.
    """
    top_k = config["top_k_context"]

    output_contexts = []
    for ctx in task.get("contexts", [])[:top_k]:
        doc_id = ctx.get("document_id")
        if doc_id in corpus:
            output_contexts.append({
                "document_id": doc_id,
                "text":        corpus[doc_id],
                "score":       ctx.get("score", 1.0)
            })

    return {
        "task_id":         task["task_id"],
        "Collection":      task.get("Collection", ""),
        "original_query":  task.get("original_query", ""),
        "rewritten_query": task.get("rewritten_query", ""),
        "contexts":        output_contexts,
        "predictions":     [{"text": gen_text}]
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate(config_path: str = "config.json"):
    """
    Full generation pipeline:
      1. Load config
      2. Load data (tasks + corpus)
      3. Load model
      4. Run inference over all tasks with contexts
      5. Normalize and save predictions to JSONL
    """
    config = load_task_c_config(config_path)
    output_path = config["paths"]["output_path"]

    print("\n--- Loading Data ---")
    query_tasks, corpus = load_data(config)
    print(f"\nTotal query tasks:     {len(query_tasks)}")
    print(f"Total corpus documents: {len(corpus)}")

    print("\n--- Loading Model ---")
    tokenizer, model, device = load_model(config)

    # Skip tasks that have no retrieval results — they cannot be answered.
    tasks_to_process = [t for t in query_tasks if t.get("contexts")]
    print(f"\n--- Generating Answers ---")
    print(f"Processing {len(tasks_to_process)} tasks "
          f"(skipping {len(query_tasks) - len(tasks_to_process)} without contexts)")

    if torch.cuda.is_available():
        print(f"Estimated time on GPU: ~{len(tasks_to_process) * 0.5 / 60:.1f} minutes")
    else:
        print(f"⚠️ Estimated time on CPU: ~{len(tasks_to_process) * 15 / 60:.1f} minutes (VERY SLOW!)")

    with open(output_path, "w") as out_f:
        for task in tqdm(tasks_to_process, desc="Generating", ncols=100):
            raw_text = run_inference(task, corpus, tokenizer, model, device, config)
            gen_text = normalize_output(raw_text, config)
            record   = build_output_record(task, corpus, gen_text, config)
            out_f.write(json.dumps(record) + "\n")

    print(f"✓ Results saved to {output_path}")

    # Release GPU VRAM after inference.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ GPU memory cleared")
        print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")


if __name__ == "__main__":
    generate()