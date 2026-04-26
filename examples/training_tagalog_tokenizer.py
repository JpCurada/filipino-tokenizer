"""
Train the Tagalog tokenizer on the Wikitext-TL39 corpus.

Download the corpus first:
    python scripts/download_corpus.py

Usage:
    python examples/training_tagalog_tokenizer.py [corpus] [--vocab-size N] [--output DIR]

Defaults:
    corpus     filipino_tokenizer/data/eval/train_corpus.txt  (Wikitext-TL39)
    vocab-size 32000
    output     demo/models/morph/
"""

import argparse
import os
import sys
import time

from filipino_tokenizer.tagalog import TagalogTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(_ROOT, "filipino_tokenizer", "data", "eval", "train_corpus.txt")
DEFAULT_OUTPUT = os.path.join(_ROOT, "demo", "models", "morph")


def main():
    parser = argparse.ArgumentParser(description="Train the Tagalog BPE tokenizer on Wikitext-TL39")
    parser.add_argument("corpus", nargs="?", default=DEFAULT_CORPUS,
                        help="Path to training corpus (default: Wikitext-TL39 local file)")
    parser.add_argument("--vocab-size", type=int, default=32_000,
                        help="BPE vocabulary size (default: 32000)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Directory to save trained tokenizer (default: demo/models/morph/)")
    args = parser.parse_args()

    if not os.path.isfile(args.corpus):
        print(f"ERROR: Corpus not found: {args.corpus}", file=sys.stderr)
        print("Run `python scripts/download_corpus.py` to download Wikitext-TL39.", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(args.corpus) / 1_000_000
    print(f"Corpus    : {args.corpus} ({size_mb:.1f} MB)")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Output    : {args.output}")
    print()

    tok = TagalogTokenizer()
    t0 = time.perf_counter()
    tok.train(args.corpus, vocab_size=args.vocab_size)
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"  Vocabulary : {len(tok.bpe.vocab):,} tokens")
    print(f"  Merges     : {len(tok.bpe.merges):,}")

    tok.save(args.output)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
