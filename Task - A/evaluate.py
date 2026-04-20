"""
evaluate.py

Compute nDCG@5, nDCG@10, Recall@5, Recall@10 for a results JSONL file
against ground-truth qrels.

Usage:
    # Evaluate one domain file
    python evaluate.py --results results/clapnq_taskA_results.jsonl \
                       --qrels   /path/to/qrels/clapnq/dev.tsv

    # Evaluate the combined submission file against all domains
    python evaluate.py --submission final_submission_taskA.jsonl \
                       --config config.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config_loader import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(qrels: dict, runs: dict, k: int) -> float:
    """Binary recall: fraction of queries where ≥1 relevant doc found in top-k."""
    hits, total = 0, 0
    for qid, rels in qrels.items():
        if qid not in runs:
            continue
        relevant = {pid for pid, r in rels.items() if r > 0}
        if relevant & set(runs[qid][:k]):
            hits += 1
        total += 1
    return hits / total if total else 0.0


def ndcg_at_k(qrels: dict, runs: dict, k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k."""
    def dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    scores = []
    for qid, rels in qrels.items():
        if qid not in runs:
            continue
        gains = [rels.get(pid, 0) for pid in runs[qid][:k]]
        ideal = sorted(rels.values(), reverse=True)[:k]
        if sum(ideal) > 0:
            scores.append(dcg(gains) / dcg(ideal))
    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_runs_from_jsonl(path: str) -> dict:
    """Read a results JSONL and return {task_id: [pid, ...]}."""
    runs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tid  = record["task_id"]
            pids = [ctx["document_id"] for ctx in record.get("contexts", [])]
            runs[tid] = pids
    return runs


def load_qrels(path: str) -> defaultdict:
    """Load a TSV qrels file and return {qid: {pid: rel}}."""
    qrels = defaultdict(dict)
    with open(path) as f:
        next(f)   # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rel = parts[0], parts[1], int(parts[2])
            qrels[qid][pid] = rel
    return qrels


def print_table(label: str, r5: float, n5: float, r10: float, n10: float,
                n_runs: int, n_qrels: int):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  {'Recall@5':<18} {r5:.4f}")
    print(f"  {'nDCG@5':<18} {n5:.4f}")
    print(f"  {'Recall@10':<18} {r10:.4f}")
    print(f"  {'nDCG@10':<18} {n10:.4f}")
    print(f"  Coverage : {n_runs}/{n_qrels} queries")
    print(f"{'─'*50}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single(results_path: str, qrels_path: str, label: str = None):
    runs  = load_runs_from_jsonl(results_path)
    qrels = load_qrels(qrels_path)
    label = label or os.path.basename(results_path)

    r5  = recall_at_k(qrels, runs, 5)
    n5  = ndcg_at_k(qrels,  runs, 5)
    r10 = recall_at_k(qrels, runs, 10)
    n10 = ndcg_at_k(qrels,  runs, 10)

    print_table(label, r5, n5, r10, n10, len(runs), len(qrels))
    return {"recall@5": r5, "ndcg@5": n5, "recall@10": r10, "ndcg@10": n10}


def evaluate_submission(submission_path: str, cfg: dict):
    """
    Evaluate a combined submission file against all domain qrels.
    Splits records by Collection field to evaluate per domain, then aggregates.
    """
    # Read all records grouped by collection
    all_runs = defaultdict(dict)
    with open(submission_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tid  = record["task_id"]
            coll = record.get("Collection", "unknown")
            pids = [ctx["document_id"] for ctx in record.get("contexts", [])]
            all_runs[coll][tid] = pids

    all_r5, all_n5, all_r10, all_n10 = [], [], [], []

    for domain in cfg["domains"]:
        qrels_path = cfg["qrels_paths"][domain]
        if not os.path.exists(qrels_path):
            print(f" Qrels not found for {domain} — skipping.")
            continue

        qrels = load_qrels(qrels_path)
        # Try both the full domain name and variants
        runs = all_runs.get(domain, all_runs.get(domain.lower(), {}))

        r5  = recall_at_k(qrels, runs, 5)
        n5  = ndcg_at_k(qrels,  runs, 5)
        r10 = recall_at_k(qrels, runs, 10)
        n10 = ndcg_at_k(qrels,  runs, 10)

        print_table(domain.upper(), r5, n5, r10, n10, len(runs), len(qrels))
        all_r5.append(r5); all_n5.append(n5)
        all_r10.append(r10); all_n10.append(n10)

    if all_n5:
        avg_r5  = sum(all_r5)  / len(all_r5)
        avg_n5  = sum(all_n5)  / len(all_n5)
        avg_r10 = sum(all_r10) / len(all_r10)
        avg_n10 = sum(all_n10) / len(all_n10)
        print_table("MACRO AVERAGE", avg_r5, avg_n5, avg_r10, avg_n10,
                    sum(len(v) for v in all_runs.values()),
                    sum(len(v) for v in all_runs.values()))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task A results.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results",    help="Single domain results JSONL.")
    group.add_argument("--submission", help="Combined submission JSONL.")
    parser.add_argument("--qrels",   help="Qrels TSV (required with --results).")
    parser.add_argument("--config",  default=None)
    args = parser.parse_args()

    if args.results:
        if not args.qrels:
            parser.error("--qrels is required when using --results.")
        evaluate_single(args.results, args.qrels)
    else:
        cfg = load_config(args.config)
        evaluate_submission(args.submission, cfg)


if __name__ == "__main__":
    main()
