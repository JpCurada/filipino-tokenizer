# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-04-27

### Fixed

- **HuggingFace wrapper — `_convert_token_to_id` never returns `None`**: After `super().__init__()`, HuggingFace wraps special tokens in `AddedToken` objects. Passing an `AddedToken` as a plain `dict` key failed the vocab lookup in some HF versions, making `unk_id` undefined and allowing `None` to propagate into padded `input_ids`, crashing batch tokenisation with `ValueError: type of None unknown`. Fixed by calling `str(self.unk_token)` before the lookup and adding a final `isinstance(result, int)` guard so the function always returns a valid integer.
- **Broader import-error handling for `transformers`**: The `try/except ImportError` around the `transformers` import was too narrow — partial installs can raise `AttributeError` or other exceptions, silently setting `PreTrainedTokenizer = object` and stripping `batch_encode_plus` and other HF methods from the class. Changed to `except Exception` and replaced the `object` fallback with a proper stub class so the MRO is never crippled.

## [0.4.0] - 2026-04-27

### Added

- **Rust BPE backend** — `filipino_tokenizer._bpe_rust.CoreBPE` is a compiled Rust extension (PyO3 + `rustc-hash`) replacing the pure-Python encode/decode loop. The greedy BPE algorithm uses a `FxHashMap` for O(1) merge-rank lookup and a reusable key buffer to eliminate per-call allocation overhead.
- **`MorphAwareBPE._init_rust()`** — builds `CoreBPE` from the current `vocab` and `merges` after `train()` or `load()`. The Python API is unchanged; the Rust layer is transparent to callers.
- **`setup.py` + `setuptools-rust`** — `pip install` now compiles the extension automatically (`setuptools-rust>=1.5.2` added to build requirements). Pre-built wheels on PyPI mean no Rust toolchain is needed for end-users.
- **Integration test suite** — `tests/test_rust_backend.py` covers extension loading, encode/decode correctness, morpheme-boundary enforcement, cache consistency, and full-sentence round-trips.

### Changed

- `MorphAwareBPE.encode()` and `decode()` delegate entirely to the Rust backend. The pure-Python `_apply_merges()` method has been removed.

## [0.3.2] - 2026-04-27

### Fixed
- **HuggingFace batch tokenization padding crash**: `TagalogHFTokenizer` now guards special-token ID resolution for custom/older vocabularies and avoids `None` IDs during padding/truncation calls (e.g., `tok(batch, padding="max_length", truncation=True)`), fixing the `ValueError: type of None unknown: <class 'NoneType'>` failure in `transformers`.
- **Safer token ID conversion**: `_convert_token_to_id()` now handles `None` input defensively and always falls back to a valid unknown-token ID.

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
