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

```bash
pip install filipino-tokenizer
```

Pre-built wheels are available for Linux, macOS, and Windows on Python 3.10–3.13 — no compiler or Rust toolchain required.

For HuggingFace Transformers integration:

```bash
pip install filipino-tokenizer[hf]
```

To install from source for development (requires Rust via [rustup.rs](https://rustup.rs)):

```bash
git clone https://github.com/JpCurada/filipino-tokenizer.git
cd filipino-tokenizer
pip install -e .
```

## Quick Start

### Use the bundled pretrained model

A 32k-vocabulary model trained on Wikitext-TL-39 ships inside the package — no download needed.

```python
from filipino_tokenizer.tagalog import TagalogTokenizer

tok = TagalogTokenizer()
tok.load_pretrained()

ids = tok.encode("Kumain siya ng pagkain.")
print(tok.decode(ids))    # kumain siya ng pagkain.
print(tok.tokenize("Kumain siya ng pagkain."))
# ['k', 'um', 'ain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']
```

### HuggingFace integration

```python
from filipino_tokenizer.tagalog import TagalogHFTokenizer

tok = TagalogHFTokenizer()   # loads bundled model
encoding = tok("Kumain siya ng pagkain.", return_tensors="pt")
```

Works directly with `Trainer`, TRL, Axolotl, LlamaFactory, and any other HuggingFace-based training pipeline.

### Train a custom model

```python
from filipino_tokenizer.tagalog import TagalogTokenizer

tok = TagalogTokenizer()
tok.train("corpus.txt", vocab_size=32000)

ids = tok.encode("Kumain siya ng pagkain.")
print(tok.decode(ids))   # kumain siya ng pagkain.

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

**Stage 3: Constrained BPE.** The `MorphAwareBPE` class runs an optimized, incremental byte-pair encoding algorithm (using doubly-linked lists and max-heaps) with one critical constraint: it never merges a pair of symbols that would cross a morpheme boundary marker (`▁`). Merges that respect this constraint are learned at training time. At inference time, the greedy BPE encoder is implemented in Rust (`_bpe_rust.CoreBPE` via PyO3) for fast, allocation-efficient encoding.

## Evaluation

We evaluated our `TagalogTokenizer` against standard industry tokenizers (GPT-4's `cl100k_base` and SentencePiece Unigram) on a 5,000-line corpus evaluation split.

```text
=======================================================================
Metric                         | Ours       | GPT-4      | SPM       
-----------------------------------------------------------------------
Total Tokens                   | 645        | 516        | 318       
Tokens per Word (Fertility)    | 2.34       | 1.87       | 1.15      
Morpheme F1 Accuracy           | 64.5%      | 20.8%      | 12.0%     
=======================================================================
```

- **Morpheme F1 Accuracy:** Our tokenizer is **3x more likely** to split Filipino words at actual linguistic boundaries than GPT-4, and **5x more likely** than SentencePiece.
- **Fertility:** Our tokenizer produces slightly more tokens per word (2.34). This is the expected trade-off: because we strictly prevent merges across morpheme boundaries, frequent but morphologically distinct parts (like `pag` and `kain`) are kept separate, rather than being memorized as a single unbroken token (`pagkain`). This ensures robust compositional understanding for AI models.

## Project Structure

```
filipino-tokenizer/
    src/
        lib.rs                  # Rust BPE backend (CoreBPE, PyO3 bindings)
    filipino_tokenizer/
        base.py                 # BaseAffixes, BaseRoots, BaseSegmenter, BaseTokenizer
        data/
            prefix_table.json       # Prefix definitions, multi-language
            suffix_table.json       # Suffix definitions
            infix_table.json        # Infix definitions
            circumfix_table.json    # Circumfix definitions
            tagalog_roots.json      # ~30k Tagalog root words
            bisaya_roots.json       # Bisaya root words
            pretrained/
                vocab.json          # Bundled 32k vocabulary (Wikitext-TL-39)
                merges.txt          # Bundled merge rules
        tagalog/
            __init__.py         # Package exports
            affixes.py          # TagalogAffixes (filters for language="Tagalog")
            roots.py            # TagalogRoots (loads tagalog_roots.json)
            phonology.py        # Nasal assimilation, suffix h-insertion
            segmenter.py        # TagalogSegmenter (multi-pass morpheme decomposition)
            bpe.py              # MorphAwareBPE (constrained BPE, delegates to Rust)
            tokenizer.py        # TagalogTokenizer (segmenter + BPE pipeline)
            hf_tokenizer.py     # TagalogHFTokenizer (PreTrainedTokenizer wrapper)
    tests/
        test_affixes.py         # Affix loading and filtering tests
        test_segmenter.py       # Morphological segmentation tests
        test_tokenizer.py       # Full pipeline tests (round-trip, consistency, efficiency)
        test_rust_backend.py    # Rust extension tests (encode/decode, morpheme boundaries)
    examples/
        training_tagalog_tokenizer.py   # End-to-end training example
    demo/
        demo_tagalog_tokenizer.ipynb    # Usage guide notebook
        tokenizer_comparisons.ipynb     # Benchmark vs GPT-4 and SentencePiece
        tokenizer_comparisons_fil.ipynb # Side-by-side comparison on Filipino sentences
        slm_tokenizer_comparison.ipynb  # SLM training metrics comparison
        slm_training_experiment.ipynb   # Full GPT-2 training experiment
    Cargo.toml                  # Rust crate configuration
    setup.py                    # setuptools-rust build hook
    pyproject.toml              # Package metadata and build system
```

## Running Tests

```bash
# All tests
python -m unittest discover tests -v

# Individual test files
python -m unittest tests.test_affixes -v
python -m unittest tests.test_segmenter -v
python -m unittest tests.test_tokenizer -v
python -m unittest tests.test_rust_backend -v

# Rust unit tests (requires cargo)
cargo test
```

## Adding a New Language

The architecture is designed to support multiple Philippine languages from the same data files. To add Bisaya, Ilokano, or another language:

1. Add entries to the JSON affix tables in `filipino_tokenizer/data/` with the appropriate `language` field.
2. Add a root word list (e.g., `filipino_tokenizer/data/bisaya_roots.json`).
3. Create `filipino_tokenizer/<language>/affixes.py` subclassing `BaseAffixes` with `super().__init__(language="<Language>")`.
4. Create a roots class subclassing `BaseRoots`.
5. Implement a segmenter subclassing `BaseSegmenter` with language-specific phonological rules.
6. Create a tokenizer class that wires the segmenter to `MorphAwareBPE`.

## References

- **Tacorda, A. J., Ignacio, M. J., Oco, N., & Roxas, R. E. (2017).** [Controlling byte pair encoding for neural machine translation](https://doi.org/10.1109/IALP.2017.8300571). *2017 International Conference on Asian Language Processing (IALP)*, 168-171. The core idea behind the boundary-constrained (Controlled) BPE approach used here.

- **Cruz, J. C. B., & Cheng, C. (2022).** [Improving Large-scale Language Models and Resources for Filipino](https://aclanthology.org/2022.lrec-1.703/). *Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC)*. Authors of key Filipino NLP datasets and benchmarks, including the TLUnified corpus.

- **Miranda, L. J. (2023).** [calamanCy: A Tagalog Natural Language Processing Toolkit](https://aclanthology.org/2023.nlposs-1.1/). *Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS)*. SpaCy-based NLP pipeline for Tagalog that informed the morphological analysis approach.

## License

MIT License. See [LICENSE](LICENSE) for details.
