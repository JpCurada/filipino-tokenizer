"""
Download the Wikitext-TL39 training corpus from HuggingFace.

Requires:
    pip install datasets

Usage:
    python scripts/download_corpus.py

Writes three files (gitignored):
    filipino_tokenizer/data/eval/train_corpus.txt
    filipino_tokenizer/data/eval/eval_corpus.txt
    filipino_tokenizer/data/eval/test_corpus.txt
"""

import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not found. Run: pip install datasets")
    sys.exit(1)

DATASET_ID = "linkanjarad/Wikitext-TL39"

SPLITS = {
    "train": "filipino_tokenizer/data/eval/train_corpus.txt",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print(f"Loading '{DATASET_ID}' from HuggingFace ...")
    ds = load_dataset(DATASET_ID)

    for split, rel_path in SPLITS.items():
        if split not in ds:
            print(f"  Split '{split}' not found in dataset, skipping.")
            continue

        out_path = os.path.join(ROOT, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        rows = ds[split]["text"]
        lines = [line for line in rows if line and line.strip()]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  [{split}] {len(lines):,} lines -> {rel_path}")

    print("Done.")


if __name__ == "__main__":
    main()
