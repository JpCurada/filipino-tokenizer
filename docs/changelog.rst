Changelog
=========

0.1.0 (2025)
------------

- Initial release.
- ``TagalogTokenizer`` — full train / encode / decode / save / load pipeline.
- ``TagalogSegmenter`` — multi-pass morphological segmenter with root validation,
  frozen-form guard, and recursive stacked-prefix support.
- ``MorphAwareBPE`` — constrained BPE with doubly-linked list + max-heap training.
- ``TagalogPhonology`` — nasal assimilation and suffix h-insertion rules.
- Data: 28,000+ Tagalog roots, 92 prefixes, 34 suffixes, 2 infixes, 30 circumfixes.
