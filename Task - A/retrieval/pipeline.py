"""
retrieval/pipeline.py

Full Task A retrieval pipeline for one domain:

    Rewritten query
        -> BM25 retrieval       (top-150)
        -> BGE dense retrieval  (top-150, FAISS IndexFlatIP)
        -> E5 dense retrieval   (top-150, FAISS IndexFlatIP, "query: " prefix)
        -> RRF fusion           (k=60, top-100)
        -> Ensemble reranking   (BGE-reranker-large x 0.40
                                  mxbai-rerank-large  x 0.35
                                  ms-marco-MiniLM     x 0.25)
        -> Top-10 passages with scores

Usage:
    # Dev mode  -- runs pipeline AND prints nDCG/Recall metrics
    python retrieval/pipeline.py --domain clapnq
    python retrieval/pipeline.py --domain all

    # Test mode -- runs pipeline, skips evaluation (no qrels needed)
    python retrieval/pipeline.py --domain clapnq --test
    python retrieval/pipeline.py --domain all    --test

    # Point at a custom test query file
    python retrieval/pipeline.py --domain clapnq --test \
        --query-path /path/to/clapnq_test_queries.jsonl
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys

import faiss
import numpy as np
import torch
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import re

def clean_passage(text: str) -> str:
    """Remove URLs and markdown-style headers (paper §3.9)."""
    text = re.sub(r'https?://\S+', '', text)          # strip URLs
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # strip headers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def min_max_normalize(x):
    x = np.array(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo == 0:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)


def reciprocal_rank_fusion(rankings, k=60):
    """
    Combine multiple ranked lists with RRF.

    Args:
        rankings : list of [(passage_id, score), ...], each sorted descending.
        k        : RRF smoothing constant (default 60).

    Returns:
        Sorted list of (passage_id, rrf_score) in descending order.
    """
    scores = {}
    for ranked_list in rankings:
        for rank, (pid, _) in enumerate(ranked_list):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_embeddings(bge_path, e5_path):
    """
    Load precomputed BGE and E5 embeddings from disk.

    Returns
    -------
    bge_emb       : np.ndarray  shape (N, dim)
    e5_emb        : np.ndarray  shape (N, dim)
    passage_ids   : list[str]   length N  -- position i matches BM25 index i
    passage_texts : list[str]   length N  -- same order; use this to build BM25
    pid_to_text   : dict        {pid: text} -- use for reranker pair lookup
    """
    print("  Loading BGE embeddings...")
    with open(bge_path, "rb") as f:
        bge_data = pickle.load(f)
    bge_emb       = bge_data["embeddings"]
    passage_ids   = bge_data["passage_ids"]    # ordered list
    passage_texts = bge_data["passage_texts"]  # ordered list, same index as passage_ids

    print("  Loading E5 embeddings...")
    with open(e5_path, "rb") as f:
        e5_emb = pickle.load(f)["embeddings"]

    assert len(passage_ids) == len(e5_emb), (
        f"BGE ({len(passage_ids)}) and E5 ({len(e5_emb)}) passage counts differ. "
        "Regenerate with preprocessing/generate_embeddings.py."
    )

    pid_to_text = dict(zip(passage_ids, passage_texts))
    print(f"  {len(passage_ids):,} passages loaded.")
    return bge_emb, e5_emb, passage_ids, passage_texts, pid_to_text


def build_faiss_index(embeddings):
    """FAISS inner-product index (cosine sim for L2-normalised vectors)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def load_queries(query_path):
    """
    Load queries from a JSONL file.

    Returns
    -------
    dict : { task_id : {"text": str, "Collection": str} }
    """
    queries = {}
    with open(query_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            tid = j["_id"]
            queries[tid] = {
                "text":       j["text"],
                "Collection": j.get("Collection", j.get("collection", "unknown")),
            }
    print(f"  {len(queries):,} queries loaded.")
    return queries


def load_qrels(qrels_path):
    """Return { qid : {pid : rel_score} }."""
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rel = parts[0], parts[1], int(parts[2])
            qrels[qid][pid] = rel
    print(f"  Qrels loaded for {len(qrels):,} queries.")
    return qrels


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_bm25(bm25_index, passage_ids, query, top_k):
    """Return top-k (passage_id, score) pairs via BM25."""
    scores = bm25_index.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(passage_ids[i], float(scores[i])) for i in top_indices]


def retrieve_dense(model, faiss_index, passage_ids, query, top_k, prefix=""):
    """
    Return top-k (passage_id, score) pairs via dense retrieval.
    E5 requires prefix="query: "; BGE uses prefix="" (empty string).
    """
    q_emb = model.encode(
        f"{prefix}{query}",
        normalize_embeddings=True,
    ).reshape(1, -1).astype("float32")
    scores, indices = faiss_index.search(q_emb, top_k)
    return [(passage_ids[i], float(s)) for i, s in zip(indices[0], scores[0])]


# ──────────────────────────────────────────────────────────────────────────────
# Reranking
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_rerank(rerankers, weights, query, candidate_pids,
                    pid_to_text, batch_size, final_top_k):
    """
    Score candidates with each cross-encoder, min-max normalise each model's
    scores independently, then compute a weighted sum.

    Returns list of (pid, score) tuples, sorted descending, length = final_top_k.
    """
    pairs    = [[query, clean_passage(pid_to_text[pid])] for pid in candidate_pids]
    combined = np.zeros(len(candidate_pids))

    for name, weight in weights.items():
        with torch.no_grad():
            raw = rerankers[name].predict(pairs, batch_size=batch_size)
        combined += weight * min_max_normalize(raw)

    ranked = sorted(range(len(combined)),
                    key=lambda i: combined[i], reverse=True)
    return [(candidate_pids[i], float(combined[i])) for i in ranked[:final_top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(qrels, runs, k):
    hits, total = 0, 0
    for qid, rels in qrels.items():
        if qid not in runs:
            continue
        relevant = {pid for pid, r in rels.items() if r > 0}
        if relevant & set(runs[qid][:k]):
            hits += 1
        total += 1
    return hits / total if total else 0.0


def ndcg_at_k(qrels, runs, k):
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


def print_metrics(runs, qrels, domain):
    r5  = recall_at_k(qrels, runs, 5)
    n5  = ndcg_at_k(qrels,  runs, 5)
    r10 = recall_at_k(qrels, runs, 10)
    n10 = ndcg_at_k(qrels,  runs, 10)
    print(f"\n{'='*55}")
    print(f"RESULTS  --  {domain.upper()}")
    print(f"{'='*55}")
    print(f"  {'Recall@5':<18} {r5:.4f}")
    print(f"  {'nDCG@5':<18} {n5:.4f}")
    print(f"  {'Recall@10':<18} {r10:.4f}")
    print(f"  {'nDCG@10':<18} {n10:.4f}")
    print(f"{'='*55}")
    print(f"  Coverage : {len(runs)}/{len(qrels)} queries")


# ──────────────────────────────────────────────────────────────────────────────
# Single-domain pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_domain(domain, cfg, shared_models=None,
               test_mode=False, custom_query_path=None):
    """
    Execute the full retrieval pipeline for one domain.

    Parameters
    ----------
    domain            : str   One of clapnq / ibmcloud / fiqa / govt.
    cfg               : dict  Resolved config dict.
    shared_models     : dict  Optional pre-loaded models to reuse across domains.
    test_mode         : bool  If True, skips qrels loading and evaluation.
                              Use this when running on the actual test set.
    custom_query_path : str   Override the query file from config (e.g. test queries).
    """
    retrieval_cfg  = cfg["retrieval"]
    rerank_cfg     = cfg["reranking"]
    model_ids      = cfg["models"]

    TOP_RETRIEVAL  = retrieval_cfg["top_retrieval"]
    TOP_FUSION     = retrieval_cfg["top_fusion"]
    FINAL_TOP_K    = retrieval_cfg["final_top_k"]
    RRF_K          = retrieval_cfg["rrf_k"]
    BATCH_SIZE     = rerank_cfg["batch_size"]
    RERANK_WEIGHTS = rerank_cfg["weights"]

    output_dir  = cfg["output"]["results_dir"]
    output_file = os.path.join(output_dir, f"{domain}_taskA_results.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    mode_label = "TEST" if test_mode else "DEV"

    print(f"\n{'='*55}")
    print(f"DOMAIN : {domain.upper()}  [{mode_label} MODE]")
    print(f"Device : {device}")
    print(f"{'='*55}")

    # ── 1. Load corpus embeddings & build indices ──────────────────────────
    print("\n[1] Loading embeddings & building indices...")
    (bge_emb, e5_emb,
     passage_ids,
     passage_texts,     # ordered list -- use this for BM25, not pid_to_text.values()
     pid_to_text) = load_embeddings(
        cfg["embedding_paths"][domain]["bge"],
        cfg["embedding_paths"][domain]["e5"],
    )
    index_bge = build_faiss_index(bge_emb)
    index_e5  = build_faiss_index(e5_emb)

    # ── 2. Build BM25 index ────────────────────────────────────────────────
    # Use passage_texts (the ordered list from the .pkl file) so that
    # BM25 internal position i is guaranteed to match passage_ids[i].
    # Do NOT use pid_to_text.values() -- dict ordering is not reliable here.
    print("[2] Building BM25 index...")
    bm25 = BM25Okapi([t.lower().split() for t in passage_texts])

    # ── 3. Load queries ────────────────────────────────────────────────────
    print("[3] Loading queries...")
    query_path = custom_query_path or cfg["query_paths"][domain]
    queries    = load_queries(query_path)

    # ── 4. Load qrels (dev mode only) ─────────────────────────────────────
    qrels = {}
    if not test_mode:
        print("     Loading qrels...")
        qrels = load_qrels(cfg["qrels_paths"][domain])

    # ── 5. Load / reuse encoder + reranker models ──────────────────────────
    if shared_models is not None:
        print("[4] Reusing shared models.")
        model_bge = shared_models["model_bge"]
        model_e5  = shared_models["model_e5"]
        rerankers = shared_models["rerankers"]
    else:
        print("[4] Loading encoder and reranker models...")
        model_bge = SentenceTransformer(model_ids["bge_encoder"]).to(device)
        model_e5  = SentenceTransformer(model_ids["e5_encoder"]).to(device)
        if device == "cuda":
            model_bge = model_bge.half()
            model_e5  = model_e5.half()

        rerankers = {
            "bge_reranker_large": CrossEncoder(model_ids["bge_reranker_large"],
                                               device=device),
            "mxbai_rerank_large": CrossEncoder(model_ids["mxbai_rerank_large"],
                                               device=device),
            "ms_marco_minilm":    CrossEncoder(model_ids["ms_marco_minilm"],
                                               device=device),
        }
        print("  Models loaded.")

    # ── 6. Main retrieval + reranking loop ─────────────────────────────────
    print(f"\n[5] Running pipeline over {len(queries):,} queries...\n")
    runs = {}

    with open(output_file, "w") as fout:
        for task_id, q_data in tqdm(queries.items(),
                                    desc=f"Processing {domain}"):
            query = q_data["text"]

            # BM25 retrieval (lexical)
            bm25_rank = retrieve_bm25(bm25, passage_ids, query, TOP_RETRIEVAL)

            # BGE dense retrieval (no prefix)
            bge_rank = retrieve_dense(model_bge, index_bge, passage_ids,
                                      query, TOP_RETRIEVAL, prefix="")

            # E5 dense retrieval ("query: " prefix is required by E5)
            e5_rank = retrieve_dense(model_e5, index_e5, passage_ids,
                                     query, TOP_RETRIEVAL, prefix="query: ")

            # RRF fusion -- top-100 candidates
            fused = reciprocal_rank_fusion(
                [bm25_rank, bge_rank, e5_rank], k=RRF_K
            )[:TOP_FUSION]
            candidate_pids = [pid for pid, _ in fused]

            # Ensemble reranking -- top-10
            reranked = ensemble_rerank(
                rerankers, RERANK_WEIGHTS, query,
                candidate_pids, pid_to_text,
                batch_size=BATCH_SIZE, final_top_k=FINAL_TOP_K,
            )

            runs[task_id] = [pid for pid, _ in reranked]

            # Write submission-format JSONL record
            record = {
                "task_id":    task_id,
                "Collection": q_data["Collection"],
                "contexts": [
                    {"document_id": pid, "score": score}
                    for pid, score in reranked
                ],
            }
            fout.write(json.dumps(record) + "\n")

    # ── 7. Evaluate (dev mode only) ────────────────────────────────────────
    if not test_mode and qrels:
        print_metrics(runs, qrels, domain)
    elif test_mode:
        print("\n[Test mode] Evaluation skipped -- no qrels for test set.")

    print(f"\nResults saved -> {output_file}")
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task A: Hybrid retrieval + ensemble reranking pipeline."
    )
    parser.add_argument(
        "--domain", required=True,
        choices=["clapnq", "ibmcloud", "fiqa", "govt", "all"],
        help="Domain to process. Use 'all' to run every domain.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Test mode: skip qrels loading and evaluation. "
            "Use this when running on the actual competition test queries."
        ),
    )
    parser.add_argument(
        "--query-path", default=None,
        help=(
            "Override the query file for a single domain. "
            "For example, point at the official test JSONL instead of dev."
        ),
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.json (optional; default location used if omitted).",
    )
    args = parser.parse_args()

    if args.query_path and args.domain == "all":
        parser.error("--query-path can only be used with a single --domain, not 'all'.")

    cfg     = load_config(args.config)
    domains = cfg["domains"] if args.domain == "all" else [args.domain]

    # Pre-load models once when running multiple domains -- saves significant time.
    if len(domains) > 1:
        print("\nPre-loading models (shared across all domains)...")
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        model_ids = cfg["models"]

        model_bge = SentenceTransformer(model_ids["bge_encoder"]).to(device)
        model_e5  = SentenceTransformer(model_ids["e5_encoder"]).to(device)
        if device == "cuda":
            model_bge = model_bge.half()
            model_e5  = model_e5.half()

        rerankers = {
            "bge_reranker_large": CrossEncoder(model_ids["bge_reranker_large"],
                                               device=device),
            "mxbai_rerank_large": CrossEncoder(model_ids["mxbai_rerank_large"],
                                               device=device),
            "ms_marco_minilm":    CrossEncoder(model_ids["ms_marco_minilm"],
                                               device=device),
        }
        shared = {"model_bge": model_bge, "model_e5": model_e5,
                  "rerankers": rerankers}
    else:
        shared = None

    for domain in domains:
        run_domain(
            domain, cfg,
            shared_models=shared,
            test_mode=args.test,
            custom_query_path=args.query_path,
        )
        clear_gpu()

    print("\nAll done.")


if __name__ == "__main__":
    main()
