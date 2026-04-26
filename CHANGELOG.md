# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
