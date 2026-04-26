# Filipino Tokenizer

A morphology-aware BPE tokenizer for Philippine languages.

Existing subword tokenizers (SentencePiece, HuggingFace BPE) treat Filipino text as raw character sequences. They have no knowledge of Filipino morphology, so they routinely split words at linguistically meaningless points. A word like *pinakamahusay* ("the best") gets fragmented into arbitrary substrings instead of its actual morphemes: *pinaka-* + *ma-* + *husay*.

This project fixes that. It combines a rule-based morphological segmenter with a constrained BPE algorithm that never merges across morpheme boundaries. The result is a tokenizer that produces fewer, more meaningful tokens for Filipino text.

## Before and After

Consider the sentence: *Kumain siya ng masarap na pagkain.*

A generic BPE tokenizer might produce:

```
["Ku", "main", " siya", " ng", " mas", "ar", "ap", " na", " pag", "ka", "in", "."]
```

This tokenizer understands that *kumain* contains the infix *-um-* and root *kain*, and that *pagkain* is prefix *pag-* plus the same root *kain*:

```
["k", "um", "ain", " ", "siya", " ", "ng", " ", "ma", "sarap", " ", "na", " ", "pag", "kain", "."]
```

The root *kain* is preserved as a single token and shared across both words. This gives downstream models a head start on understanding Filipino word formation.

## Installation

Clone the repository and create a virtual environment. No external dependencies are required for the core library.

```bash
git clone https://github.com/JpCurada/filipino-tokenizer.git
cd filipino-tokenizer
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/macOS
```

To run tests, install pytest:

```bash
pip install pytest
```

## Quick Start

```python
import os, tempfile
from src.tagalog import TagalogTokenizer

# Write a small training corpus
corpus_text = """
Kumain siya ng pagkain sa hapagkainan.
Maganda ang panahon ngayon kaya lumabas kami.
Nagluluto ang nanay ng masarap na adobo para sa pamilya.
"""
tmpdir = tempfile.mkdtemp()
corpus_path = os.path.join(tmpdir, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write(corpus_text)

# Train
tok = TagalogTokenizer()
tok.train(corpus_path, vocab_size=500)

# Encode and decode
ids = tok.encode("Kumain siya ng pagkain.")
text = tok.decode(ids)
print(text)  # kumain siya ng pagkain.

# Inspect subword tokens
tokens = tok.tokenize("Kumain siya ng pagkain.")
print(tokens)  # ['k', 'um', 'ain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']

# Save and reload
tok.save("my_tokenizer/")
tok2 = TagalogTokenizer()
tok2.load("my_tokenizer/")
```

## How It Works

The tokenizer is a three-stage pipeline.

**Stage 1: Affix Tables.** Four JSON files in `data/` define every known Filipino prefix, suffix, infix, and circumfix. Each entry is tagged by language (Tagalog, Cebuano, etc.), so the same data files support multiple Philippine languages. Prefixes are sorted longest-first for greedy matching.

**Stage 2: Morphological Segmenter.** The `TagalogSegmenter` decomposes a word into its constituent morphemes using a multi-pass algorithm:

1. Check for frozen/lexicalized forms (e.g., *pangalan* is a word, not *pang-* + *alan*).
2. Try circumfix detection (prefix + suffix pairs like *ka- -han*).
3. Strip prefixes, longest match first, with recursion for stacked prefixes.
4. Detect infixes (*-um-* and *-in-* after the first consonant).
5. Strip suffixes, applying phonological rules (*-an* becomes *-han* after vowels).
6. Validate every candidate root against a dictionary of 30,000+ Tagalog roots.

If no valid segmentation is found, the word is returned whole.

**Stage 3: Constrained BPE.** The `MorphAwareBPE` class runs standard byte-pair encoding with one critical constraint: it never merges a pair of symbols that would cross a morpheme boundary marker. This means learned subword units always stay within a single morpheme. The approach follows the Constrained BPE (CBPE) method described by Tacorda et al.

## Project Structure

```
filipino-tokenizer/
    data/
        prefix_table.json       # Prefix definitions, multi-language
        suffix_table.json       # Suffix definitions
        infix_table.json        # Infix definitions
        circumfix_table.json    # Circumfix definitions
        tagalog_roots.json      # ~30k Tagalog root words
        bisaya_roots.json       # Bisaya root words
    src/
        base.py                 # BaseAffixes, BaseRoots, BaseSegmenter, BaseTokenizer
        tagalog/
            __init__.py         # Package exports
            affixes.py          # TagalogAffixes (filters for language="Tagalog")
            roots.py            # TagalogRoots (loads tagalog_roots.json)
            phonology.py        # Nasal assimilation, suffix h-insertion
            segmenter.py        # TagalogSegmenter (multi-pass morpheme decomposition)
            bpe.py              # MorphAwareBPE (constrained BPE, no cross-boundary merges)
            tokenizer.py        # TagalogTokenizer (segmenter + BPE pipeline)
    tests/
        test_affixes.py         # Affix loading and filtering tests
        test_segmenter.py       # Morphological segmentation tests
        test_tokenizer.py       # Full pipeline tests (round-trip, consistency, efficiency)
    examples/
        training_tagalog_tokenizer.py   # End-to-end training example
    demo/
        demo_tagalog_tokenizer.ipynb    # Jupyter notebook demo
```

## Running Tests

```bash
# All tests
python -m unittest discover tests -v

# Individual test files
python -m unittest tests.test_affixes -v
python -m unittest tests.test_segmenter -v
python -m unittest tests.test_tokenizer -v
```

## Adding a New Language

The architecture is designed to support multiple Philippine languages from the same data files. To add Bisaya, Ilokano, or another language:

1. Add entries to the JSON affix tables in `data/` with the appropriate `language` field.
2. Add a root word list (e.g., `data/bisaya_roots.json`).
3. Create `src/<language>/affixes.py` subclassing `BaseAffixes` with `super().__init__(language="<Language>")`.
4. Create a roots class subclassing `BaseRoots`.
5. Implement a segmenter subclassing `BaseSegmenter` with language-specific phonological rules.
6. Create a tokenizer class that wires the segmenter to `MorphAwareBPE`.

The Cebuano affixes class already exists at `src/cebuano/affixes.py` as a starting point.

## References

- **Tacorda, Livelo, Ong, and Cheng (2024).** Constraining Byte Pair Encoding (CBPE) to improve morphological segmentation for Filipino tokenizers. The core idea behind the boundary-constrained BPE approach used here.

- **Cruz, J.P. and Cheng, C. (2022).** Improving Large-scale Language Models and Resources for Filipino. Authors of key Filipino NLP datasets and benchmarks.

- **Miranda, L.J. (2023).** calamanCy: A Tagalog Natural Language Processing Toolkit. SpaCy-based NLP pipeline for Tagalog that informed the morphological analysis approach.

- **Affix and root data** compiled from Wiktionary's Philippine language entries.

## License

MIT License. See [LICENSE](LICENSE) for details.
