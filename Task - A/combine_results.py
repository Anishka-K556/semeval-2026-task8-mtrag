"""
combine_results.py

Merges the per-domain results files produced by retrieval/pipeline.py
into a single submission JSONL file.

Usage:
    python combine_results.py
    python combine_results.py --config path/to/config.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-domain results into one submission file."
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir     = cfg["output"]["results_dir"]
    submission_file = cfg["output"]["submission_file"]
    domains         = cfg["domains"]

    os.makedirs(os.path.dirname(submission_file), exist_ok=True)

    total = 0
    with open(submission_file, "w") as fout:
        for domain in domains:
            domain_file = os.path.join(results_dir,
                                       f"{domain}_taskA_results.jsonl")
            if not os.path.exists(domain_file):
                print(f" Missing: {domain_file} — skipping.")
                continue

            count = 0
            with open(domain_file) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    # Enforce exactly 10 contexts
                    record["contexts"] = record["contexts"][:10]
                    fout.write(json.dumps(record) + "\n")
                    count += 1

            print(f"  {domain:<12} → {count:>5} records")
            total += count

    print(f"\n Submission file: {submission_file}")
    print(f"   Total records  : {total}")


if __name__ == "__main__":
    main()
