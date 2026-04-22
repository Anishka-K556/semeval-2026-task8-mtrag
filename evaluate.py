"""
evaluate.py

Responsibilities:
  - Load predictions from the output JSONL file.
  - Compute ROUGE-L F1 over records that contain a `reference_answer` field
    (dev / validation splits only).
  - Display sample predictions for a quick sanity check.

Note (Assumption A7):
  ROUGE-L is computed only when a `reference_answer` field exists in the output.
  Test-set submissions have no ground truth and will report
  "No valid reference answers available."
"""

import json
from rouge_score import rouge_scorer

from postprocessing.normalize import safe_normalize
from utils import load_task_c_config


# =============================================================================
# RESULTS LOADING
# =============================================================================

def load_results(output_path: str) -> list:
    """
    Read all JSONL records from the predictions file.

    Args:
        output_path (str): Path to the JSONL predictions file.

    Returns:
        list[dict]: One dict per prediction record.
    """
    results = []
    with open(output_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_rouge_l(results: list) -> list:
    """
    Compute ROUGE-L F1 for each result that has a `reference_answer` field.

    Args:
        results (list[dict]): Loaded prediction records.

    Returns:
        list[float]: ROUGE-L F1 scores for records with a reference answer.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for r in results:
        if "reference_answer" not in r:
            continue   # Test split — no ground truth available.

        ref = safe_normalize(r["reference_answer"])
        gen = (
            safe_normalize(r["predictions"][0].get("text", ""))
            if r.get("predictions") else ""
        )

        # Only score when both strings are non-empty to avoid degenerate scores.
        if ref and gen:
            scores.append(
                scorer.score(ref, gen)["rougeL"].fmeasure
            )

    return scores


# =============================================================================
# SAMPLE DISPLAY
# =============================================================================

def display_samples(results: list, n: int = 3):
    """
    Print the first `n` prediction records for a quick sanity check.

    Args:
        results (list[dict]): Loaded prediction records.
        n       (int):        Number of samples to display.
    """
    print("\n--- Sample Results ---")
    for i, r in enumerate(results[:n]):
        print(f"\nTask {i+1} (ID: {r.get('task_id', 'N/A')}):")
        print(f"Query:     {r.get('rewritten_query', r.get('original_query', 'N/A'))}")
        gen = r["predictions"][0].get("text", "") if r.get("predictions") else ""
        print(f"Generated: {gen}")
        if "reference_answer" in r:
            print(f"Reference: {r['reference_answer'][:150]}...")


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def evaluate(config_path: str = "config.json"):
    """
    Full evaluation pipeline:
      1. Load config.
      2. Load predictions from the output JSONL file.
      3. Compute and print mean ROUGE-L F1 (dev set only).
      4. Display sample predictions.
    """
    config      = load_task_c_config(config_path)
    output_path = config["paths"]["output_path"]

    print("\n--- Evaluating Results ---")
    results = load_results(output_path)

    scores = compute_rouge_l(results)

    if scores:
        print(f"Mean ROUGE-L: {sum(scores) / len(scores):.4f}")
    else:
        print("No valid reference answers available (test set).")

    print("=" * 60)

    display_samples(results)


if __name__ == "__main__":
    evaluate()