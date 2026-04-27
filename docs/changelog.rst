Changelog
=========

0.4.1 (2026-04-27)
-------------------

Fixed
~~~~~

- **HuggingFace wrapper — ``_convert_token_to_id`` never returns ``None``** —
  After ``super().__init__()``, HuggingFace wraps special tokens in ``AddedToken``
  objects.  Passing an ``AddedToken`` as a plain ``dict`` key failed the vocab lookup
  in some HF versions, allowing ``None`` to propagate into padded ``input_ids`` and
  crashing batch tokenisation with:

  ``ValueError: type of None unknown: <class 'NoneType'>``

  Fixed by using ``str(self.unk_token)`` before the vocab lookup and adding a final
  ``isinstance(result, int)`` guard so the function always returns a valid integer.

- **Broader import-error handling for ``transformers``** — The ``except ImportError``
  around the ``transformers`` import was too narrow; partial installs can raise
  ``AttributeError`` or other exceptions, silently stripping ``batch_encode_plus`` and
  other HF methods.  Changed to ``except Exception`` with a proper stub-class fallback
  so the MRO is never crippled.

0.4.0 (2026-04-27)
-------------------

Added
~~~~~

- **Rust BPE backend** — ``filipino_tokenizer._bpe_rust.CoreBPE`` is a compiled
  Rust extension (PyO3 + ``rustc-hash``) that replaces the pure-Python encode/decode
  loop. The greedy O(n²) BPE algorithm, tab-separated merge-rank lookup via
  ``FxHashMap``, and a reusable key buffer eliminate per-call allocation overhead.

- **``MorphAwareBPE._init_rust()``** — builds the ``CoreBPE`` from the current
  ``vocab`` and ``merges`` after ``train()`` or ``load()``. The Python object retains
  its ``encode``/``decode`` API; the Rust layer is transparent to callers.

- **``setup.py`` + ``setuptools-rust``** — ``pip install`` now compiles the Rust
  extension automatically (``setuptools-rust>=1.5.2`` added to build requirements).

- **Integration test suite** — ``tests/test_rust_backend.py`` covers extension
  loading, encode/decode correctness, morpheme-boundary enforcement, encode-cache
  consistency, and full-sentence round-trips.

Changed
~~~~~~~

- ``MorphAwareBPE.encode()`` and ``decode()`` delegate entirely to the Rust backend.
  The pure-Python ``_apply_merges()`` method has been removed.

Breaking changes
~~~~~~~~~~~~~~~~

- **Rust is required to build the package.** Installing without a Rust toolchain will
  fail at the compilation step. See the :doc:`installation` page for setup instructions.

0.3.2 (2026-04-27)
-------------------

Fixed
~~~~~

- **HuggingFace batch padding crash** — ``TagalogHFTokenizer`` now ensures
  special tokens resolve to valid IDs even for custom/older vocabularies.
  This prevents ``None`` from leaking into padded batches and fixes:

  ``ValueError: type of None unknown: <class 'NoneType'>``

- **Defensive ID conversion** — ``_convert_token_to_id()`` now handles
  ``None`` safely and always falls back to a valid unknown-token ID.

0.3.0 (2026-04-26)
-------------------

Added
~~~~~

- **Bundled pretrained model** — ``vocab.json`` + ``merges.txt`` (32k vocabulary trained on
  Wikitext-TL-39) are now shipped inside the package. No separate download or path needed.

- **``TagalogTokenizer.load_pretrained()``** — loads the bundled model in one call:

  .. code-block:: python

     from filipino_tokenizer.tagalog import TagalogTokenizer

     tok = TagalogTokenizer()
     tok.load_pretrained()
     ids = tok.encode("Kumain siya ng pagkain.")

- **Zero-argument ``TagalogHFTokenizer()``** — ``vocab_file`` is now optional.
  Calling with no arguments loads the bundled pretrained model automatically:

  .. code-block:: python

     from filipino_tokenizer.tagalog import TagalogHFTokenizer

     tok = TagalogHFTokenizer()   # loads bundled 32k model
     encoding = tok("Kumain siya ng pagkain.", return_tensors="pt")

Changed
~~~~~~~

- ``TagalogHFTokenizer.__init__`` — ``vocab_file`` and ``merges_file`` are now optional
  (default ``None``). Existing code passing explicit file paths continues to work unchanged.

0.2.0 (2026-04-26)
-------------------

Added
~~~~~

- **HuggingFace integration** — ``TagalogHFTokenizer`` wraps the tokenizer behind the
  ``PreTrainedTokenizer`` interface, making it compatible with ``Trainer``, TRL, Axolotl,
  LlamaFactory, and any other HuggingFace-based training pipeline.

  .. code-block:: bash

     pip install filipino-tokenizer[hf]

  .. code-block:: python

     from filipino_tokenizer.tagalog import TagalogHFTokenizer

     tok = TagalogHFTokenizer.from_pretrained("path/to/model/")
     encoding = tok("Kumain siya ng pagkain.", return_tensors="pt")

- **Corpus download script** — ``scripts/download_corpus.py`` fetches
  `Wikitext-TL-39 <https://huggingface.co/datasets/linkanjarad/Wikitext-TL39>`_
  (~1.5M sentences) from HuggingFace Hub.

- **Pretrained model** — 32,000-token vocabulary trained on Wikitext-TL-39 in ~2.6 minutes,
  included in ``demo/models/morph/``.

- **Training progress reporting** — live ``stderr`` progress bars for both
  morphological segmentation and BPE merge phases (every 5%).

- **Usage guide notebook** — ``demo/demo_tagalog_tokenizer.ipynb`` covers all library
  features end-to-end: training, encode/decode, segmentation, save/reload,
  HuggingFace integration, and SLM/LLM training setup.

- **Optional dependency group** — ``pip install filipino-tokenizer[hf]``
  installs ``transformers>=4.30``.

Changed
~~~~~~~

- ``examples/training_tagalog_tokenizer.py`` rewritten to focus on Wikitext-TL-39.
  Accepts ``--vocab-size`` and ``--output`` CLI arguments.

0.1.0 (2026-04-26)
-------------------

- Initial release.
- ``TagalogTokenizer`` — full train / encode / decode / save / load pipeline.
- ``TagalogSegmenter`` — multi-pass morphological segmenter with root validation,
  frozen-form guard, and recursive stacked-prefix support.
- ``MorphAwareBPE`` — constrained BPE with doubly-linked list + max-heap training.
- ``TagalogPhonology`` — nasal assimilation and suffix h-insertion rules.
- Data: 28,000+ Tagalog roots, 92 prefixes, 34 suffixes, 2 infixes, 30 circumfixes.
