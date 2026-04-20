"""
preprocessing/generate_embeddings.py

One-time script: encodes the corpus for a given domain using both
BGE-base and E5-base and saves the embeddings as .pkl files.

Usage:
    python preprocessing/generate_embeddings.py --domain clapnq
    python preprocessing/generate_embeddings.py --domain ibmcloud
    python preprocessing/generate_embeddings.py --domain fiqa
    python preprocessing/generate_embeddings.py --domain govt

Each run saves two files:
    Embeddings/<domain>_bge_base.pkl
    Embeddings/<domain>_e5_base.pkl

Run this ONCE per domain before running the main pipeline.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus(corpus_path: str):
    """Load passage texts and IDs from a JSONL corpus file."""
    passage_ids, passage_texts = [], []
    with open(corpus_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            # Support both 'id' and '_id' field names
            pid = obj.get("_id", obj.get("id"))
            passage_ids.append(pid)
            passage_texts.append(obj["text"])
    print(f"  Loaded {len(passage_ids):,} passages from {corpus_path}")
    return passage_ids, passage_texts


def encode_corpus(model: SentenceTransformer, texts: list, batch_size: int,
                  device: str, prefix: str = "") -> np.ndarray:
    """Encode a list of texts with optional prefix (required by E5)."""
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    return embeddings


def save_embeddings(path: str, embeddings: np.ndarray,
                    passage_ids: list, passage_texts: list,
                    model_name: str, prefix: str = ""):
    """Persist embeddings and metadata to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "embeddings":    embeddings,
        "passage_ids":   passage_ids,
        "passage_texts": passage_texts,
        "model_name":    model_name,
        "normalized":    True,
        "prefix_used":   prefix,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Saved {path}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate corpus embeddings for one domain.")
    parser.add_argument("--domain", required=True,
                        choices=["clapnq", "ibmcloud", "fiqa", "govt"],
                        help="Domain to embed.")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (optional).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip encoding if .pkl already exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain = args.domain

    corpus_path  = cfg["corpus_paths"][domain]
    bge_out_path = cfg["embedding_paths"][domain]["bge"]
    e5_out_path  = cfg["embedding_paths"][domain]["e5"]
    bge_model_id = cfg["models"]["bge_encoder"]
    e5_model_id  = cfg["models"]["e5_encoder"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Domain : {domain.upper()}")
    print(f"Device : {device}")
    print(f"{'='*60}\n")

    # Determine batch size based on GPU memory
    if device == "cuda":
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 128 if gpu_gb > 15 else (64 if gpu_gb > 10 else 32)
    else:
        batch_size = 8

    # Load corpus once
    print("Loading corpus...")
    passage_ids, passage_texts = load_corpus(corpus_path)

    # ── BGE embeddings ────────────────────────────────────────────────────────
    if args.skip_existing and os.path.exists(bge_out_path):
        print(f"\nSkipping BGE (already exists): {bge_out_path}")
    else:
        print(f"\n[1/2] Encoding with BGE ({bge_model_id})...")
        model_bge = SentenceTransformer(bge_model_id).to(device)
        bge_emb = encode_corpus(model_bge, passage_texts,
                                batch_size=batch_size, device=device, prefix="")
        print(f"  Shape: {bge_emb.shape}")
        save_embeddings(bge_out_path, bge_emb, passage_ids, passage_texts,
                        model_name=bge_model_id, prefix="")
        del model_bge, bge_emb
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── E5 embeddings ─────────────────────────────────────────────────────────
    if args.skip_existing and os.path.exists(e5_out_path):
        print(f"\nSkipping E5 (already exists): {e5_out_path}")
    else:
        print(f"\n[2/2] Encoding with E5 ({e5_model_id})...")
        model_e5 = SentenceTransformer(e5_model_id).to(device)
        e5_emb = encode_corpus(model_e5, passage_texts,
                               batch_size=batch_size, device=device,
                               prefix="passage: ")
        print(f"  Shape: {e5_emb.shape}")
        save_embeddings(e5_out_path, e5_emb, passage_ids, passage_texts,
                        model_name=e5_model_id, prefix="passage: ")
        del model_e5, e5_emb
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n Embeddings ready for domain: {domain}")


if __name__ == "__main__":
    main()
