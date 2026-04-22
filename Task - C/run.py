"""
run.py — Top-level entry point for the Task-C pipeline.

Usage:
    python run.py                    # uses default config.json
    python run.py --config my.json   # custom config path
    python run.py --eval-only        # skip generation, evaluate existing output
"""

import argparse
from generation.generate import generate
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Task-C RAG Pipeline")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the config JSON file (default: config.json)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip generation and only run evaluation on existing output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.eval_only:
        print("=" * 60)
        print("STEP 1/2 — GENERATION")
        print("=" * 60)
        generate(config_path=args.config)

    print("\n" + "=" * 60)
    print("STEP 2/2 — EVALUATION")
    print("=" * 60)
    evaluate(config_path=args.config)


if __name__ == "__main__":
    main()
