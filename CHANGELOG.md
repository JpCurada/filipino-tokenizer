# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-26

### Added
- **Bundled pretrained model**: `vocab.json` + `merges.txt` (32k vocabulary, trained on Wikitext-TL-39) are now shipped inside the package at `filipino_tokenizer/data/pretrained/`. No separate download or path needed.
- **`TagalogTokenizer.load_pretrained()`**: Loads the bundled model in one call — works after `pip install`, on Kaggle, Colab, or any environment.
- **Zero-argument `TagalogHFTokenizer()`**: `vocab_file` is now optional. Calling `TagalogHFTokenizer()` with no arguments loads the bundled pretrained model automatically.

### Changed
- `TagalogHFTokenizer.__init__` signature: `vocab_file` and `merges_file` are now optional (`None` by default). Existing code passing explicit paths continues to work unchanged.

## [0.2.0] - 2026-04-26

### Added
- **HuggingFace Integration**: `TagalogHFTokenizer(PreTrainedTokenizer)` — wraps the tokenizer behind the HuggingFace `transformers` interface. Compatible with `Trainer`, TRL, Axolotl, LlamaFactory, and any other HF-based training pipeline. Install with `pip install filipino-tokenizer[hf]`.
- **Corpus download script**: `scripts/download_corpus.py` — fetches the [Wikitext-TL-39](https://huggingface.co/datasets/linkanjarad/Wikitext-TL39) dataset (~1.5M sentences) from HuggingFace and writes it to the local eval directory.
- **Pretrained model**: Tokenizer trained on Wikitext-TL-39 (32,000 token vocabulary, 31,900 merge rules) included in `demo/models/morph/`. Trains in ~2.6 minutes on a modern CPU.
- **Training progress reporting**: Live `stderr` progress bars for both the morphological segmentation phase and the BPE merge loop (every 5%).
- **Usage guide notebook**: `demo/demo_tagalog_tokenizer.ipynb` — complete end-to-end walkthrough covering all library features: training, encode/decode, morphological segmentation, save/reload, HuggingFace integration, and SLM/LLM training setup.
- **Optional dependency group**: `pip install filipino-tokenizer[hf]` installs `transformers>=4.30`.

### Changed
- `examples/training_tagalog_tokenizer.py` rewritten to focus on Wikitext-TL-39. Accepts `--vocab-size` and `--output` CLI arguments. Falls back gracefully with instructions if corpus is not downloaded.

## [0.1.0] - 2026-04-26

### Added
- **Core Library Architecture**: Initialized `filipino_tokenizer` structure with base classes for extensible Philippine language tokenization.
- **Data Layers**: Integrated Tagalog and Cebuano affix tables (`prefix_table.json`, `suffix_table.json`, `infix_table.json`, `circumfix_table.json`) and a comprehensive Tagalog root dictionary (`tagalog_roots.json`).
- **Morphological Segmenter**: Implemented multi-pass `TagalogSegmenter` supporting circumfixes, stacked prefixes, infixes, and phonological changes (nasal assimilation).
- **Constrained BPE (CBPE)**: Implemented an optimized, highly performant `MorphAwareBPE` using a doubly-linked list and max-heap logic to perform O(N log N) merges strictly adhering to morpheme boundary markers (`▁`).
- **Caching**: Added `_segment_cache` and `_encode_cache` to drastically reduce evaluation latency.
- **Documentation**: Initialized Sphinx documentation with autodoc capabilities and GitHub Actions deployment.

### Fixed
- Cache mutability issues in `MorphAwareBPE` causing token sequence pollution.
- Reinstated the internal handling of boundary markers during encoding for round-trip linguistic fidelity.
