"""
query_rewriting/rewrite.py

Stage 1 of Subtask A: Convert multi-turn conversational histories into
standalone, self-contained queries using Gemma-2-2B-IT with few-shot
prompting. No fine-tuning is performed -- the model is used purely
through prompt engineering.

What this does:
    - Loads the raw conversational query JSONL (last-turn format)
    - For each conversation, builds a few-shot prompt showing the full
      conversation history and asks Gemma to rewrite the final turn into
      a standalone query that resolves coreferences and implicit context
    - Writes the rewritten queries to a new JSONL file that the retrieval
      pipeline can consume directly

Usage:
    python query_rewriting/rewrite.py --domain clapnq
    python query_rewriting/rewrite.py --domain ibmcloud
    python query_rewriting/rewrite.py --domain fiqa
    python query_rewriting/rewrite.py --domain govt
    python query_rewriting/rewrite.py --domain all

Output:
    Writes <domain>_rewrite.jsonl next to the original query files.
    The output format is identical to the input format but with the
    "text" field replaced by the rewritten query.

Notes:
    - Requires a GPU with at least 8 GB VRAM. Falls back to CPU (very slow).
    - The model is loaded in 4-bit quantisation by default to save memory.
      Set --no-quantize to load in full precision.
    - If the model fails to rewrite a query (empty output, timeout, etc.)
      the original last-turn query is kept as a fallback.
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Context building
# ──────────────────────────────────────────────────────────────────────────────

def build_conversation_context(record: dict, max_history: int = 3):
    """
    Extract the current (last) user query and build a plain-text context
    string from the preceding turns.

    Supports two input formats:
      - SemEval format : record["input"] is a list of {"speaker", "text"} dicts
      - Pipeline format: record["history"] is a list of {"role", "text"} dicts
                         and record["text"] is the current query

    Returns
    -------
    current_query : str   – the last user turn to be rewritten
    context       : str   – formatted prior turns (empty string if turn 1)
    """
    # ── SemEval format ────────────────────────────────────────────────────
    if "input" in record:
        conversation = record["input"]
        user_turns = [t["text"] for t in conversation if t["speaker"] == "user"]

        if not user_turns:
            raise ValueError(
                f"No user turns found for task_id {record.get('task_id', 'unknown')}"
            )

        current_query = user_turns[-1]

        if len(conversation) <= 1:
            return current_query, ""

        context_parts = []
        for turn in conversation[:-1]:
            speaker = "User" if turn["speaker"] == "user" else "Assistant"
            context_parts.append(f"{speaker}: {turn['text']}")

        context = "\n".join(context_parts[-(max_history * 2):])
        return current_query, context

    # ── Pipeline / JSONL format ───────────────────────────────────────────
    current_query = record["text"]
    history_turns = record.get("history", [])

    if not history_turns:
        return current_query, ""

    context_parts = []
    for turn in history_turns:
        role = turn.get("role", turn.get("speaker", "user"))
        text = turn.get("text", turn.get("content", ""))
        speaker = "User" if role == "user" else "Assistant"
        context_parts.append(f"{speaker}: {text}")

    context = "\n".join(context_parts[-(max_history * 2):])
    return current_query, context


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(current_query: str, context: str) -> str:
    """
    Build the Gemma-2 chat-template prompt for query rewriting.
    Uses the same few-shot examples and rules as the main notebook.
    """
    prompt = f"""<start_of_turn>user
You are an expert at query de-contextualization. Your task is to rewrite the "Latest Query" into a standalone version that retains its original meaning but resolves all pronouns and implicit references using the "Context".

Rules:
1. If the query is already standalone, do not change it.
2. Never change the intent of the question.
3. If the user asks "Are there more?" or "Is that all?", rewrite it to refer to the specific topic in the context.
4. Strictly remove conversational phrases like "You said before", "I mean", or "Actually".
5. Retain the specific keywords like Countries, Names, Continents in the query


Examples:
Context: User asks how to bake a cake. Assistant provides 3 steps.
Latest Query: Are those the only steps?
Rewritten Query: Are those 3 steps the only steps required to bake a cake?

Context: Discussion about IBM Watson intents. Assistant explains what they are.
Latest Query: How is it created?
Rewritten Query: How is an intent created in IBM Watson?

Actual Task:
Context: {context}
Latest Query: {current_query}
Rewritten Query:<end_of_turn>
<start_of_turn>model
"""
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Rewriting
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rewrite_query(current_query: str, context: str,
                  model, tokenizer, device: str,
                  max_new_tokens: int = 50) -> str:
    """
    Rewrite a single query using the Gemma-2 model.

    Decodes only the newly generated tokens (not the prompt) and returns
    the first line of output, which is the rewritten query.
    Falls back to the original query if the output is empty.
    """
    prompt = build_prompt(current_query, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens (exclude prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    rewritten = generated_text.strip().split("\n")[0]

    # Fallback: keep the original if the model returned nothing
    if not rewritten:
        rewritten = current_query

    return rewritten


def rewrite_queries(raw_query_path: str, output_path: str,
                    model, tokenizer, device: str,
                    max_new_tokens: int = 50):
    """
    Iterate over a query JSONL file, rewrite each query with Gemma,
    and write results to output_path.

    Supports both the SemEval task format (with "input" / "task_id" fields)
    and the pipeline JSONL format (with "text" / "history" / "_id" fields).
    The output always uses the pipeline format so the retrieval stage can
    consume it directly.
    """
    results = []

    with open(raw_query_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in tqdm(records, desc=f"Rewriting {os.path.basename(raw_query_path)}"):
        try:
            current_query, context = build_conversation_context(record)
        except ValueError as e:
            print(f"\n  Warning: {e}")
            continue

        try:
            rewritten = rewrite_query(
                current_query, context, model, tokenizer, device,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            record_id = record.get("_id", record.get("task_id", "?"))
            print(f"  Warning: rewriting failed for {record_id}: {e}")
            rewritten = current_query

        # Build output record in pipeline format, preserving original fields
        out_record = dict(record)
        out_record["original_query"] = current_query
        out_record["text"]           = rewritten   # overwrite with rewritten
        out_record["has_context"]    = bool(context)
        out_record["context_length"] = len(context.split("\n")) if context else 0
        results.append(out_record)

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  Saved {len(results)} rewritten queries -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Maps domain name -> raw query filename suffix
# (the lastturn files are the input; we produce the rewrite files)
LASTTURN_SUFFIXES = {
    "clapnq":   "clapnq_lastturn.jsonl",
    "ibmcloud": "cloud_lastturn.jsonl",
    "fiqa":     "fiqa_lastturn.jsonl",
    "govt":     "govt_lastturn.jsonl",
}

REWRITE_SUFFIXES = {
    "clapnq":   "clapnq_rewrite.jsonl",
    "ibmcloud": "cloud_rewrite.jsonl",
    "fiqa":     "fiqa_rewrite.jsonl",
    "govt":     "govt_rewrite.jsonl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Rewrite conversational queries with Gemma-2-2B-IT."
    )
    parser.add_argument(
        "--domain", required=True,
        choices=["clapnq", "ibmcloud", "fiqa", "govt", "all"],
    )
    parser.add_argument(
        "--model-id", default="google/gemma-2-2b-it",
        help="HuggingFace model ID for the rewriting model.",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Load the model in full precision (requires more VRAM).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Maximum tokens to generate for each rewritten query.",
    )
    parser.add_argument(
        "--config", default=None,
    )
    args = parser.parse_args()

    cfg     = load_config(args.config)
    base    = cfg["base_path"]
    domains = cfg["domains"] if args.domain == "all" else [args.domain]

    # ── Load Gemma ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {args.model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if not args.no_quantize and device == "cuda":
        # 4-bit quantisation -- reduces VRAM from ~5 GB to ~2.5 GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,   # matches notebook
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)

    model.eval()
    print("  Model loaded.\n")

    # ── Rewrite per domain ────────────────────────────────────────────────
    queries_dir = os.path.join(base, "queries")

    for domain in domains:
        raw_path = os.path.join(queries_dir, LASTTURN_SUFFIXES[domain])
        out_path = os.path.join(queries_dir, REWRITE_SUFFIXES[domain])

        if not os.path.exists(raw_path):
            print(f"WARNING: Input file not found, skipping {domain}: {raw_path}")
            continue

        print(f"\n[{domain.upper()}]")
        rewrite_queries(raw_path, out_path, model, tokenizer, device,
                        max_new_tokens=args.max_new_tokens)

    print("\nQuery rewriting complete.")


if __name__ == "__main__":
    main()
