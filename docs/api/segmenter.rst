TagalogSegmenter
================

.. autoclass:: filipino_tokenizer.tagalog.segmenter.TagalogSegmenter
   :members:
   :undoc-members:
   :show-inheritance:

----

Method reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Method
     - Signature
     - Description
   * - ``segment``
     - ``(word) → list[str]``
     - Decompose a single word into morphemes.
   * - ``segment_text``
     - ``(text) → list[str]``
     - Split text on whitespace/punctuation, then segment each word.

----

Segmentation pass order
-----------------------

1. **Frozen-form guard** — words whose affix analysis is blocked by
   identical-definition duplicates in the root dictionary.
2. **Circumfix detection** — ka- -han, pag- -an, etc.
3. **Prefix stripping** — longest-match-first, recursive for stacked prefixes
   (depth limit: 3).
4. **Infix detection** — ``-um-`` and ``-in-`` after first consonant.
5. **Suffix stripping** — ``-an``/``-han``, ``-in``/``-hin`` phonology variants.
6. **Fallback** — return ``[word]`` unsegmented.

Root validation is applied at every pass: a candidate root must be ≥ 4 characters
and present in ``tagalog_roots.json``.

----

Examples
--------

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogSegmenter

   seg = TagalogSegmenter()

   # Infix
   seg.segment("kumain")          # ['um', 'kain']
   seg.segment("kinain")          # ['in', 'kain']

   # Prefix
   seg.segment("pagkain")         # ['pag', 'kain']
   seg.segment("maganda")         # ['ma', 'ganda']

   # Circumfix
   seg.segment("pagkainan")       # ['pag', 'kain', 'an']
   seg.segment("kasiyahan")       # ['ka', 'siya', 'han']

   # Stacked prefixes
   seg.segment("pinakamahusay")   # ['pinaka', 'ma', 'husay']

   # Frozen form (identical definitions for whole word and stripped root)
   seg.segment("pangalan")        # ['pangalan']

   # Loan word / no valid root found
   seg.segment("computer")        # ['computer']

   # Empty input
   seg.segment("")                # []

   # Case-insensitive
   seg.segment("KUMAIN") == seg.segment("kumain")   # True

   # Full sentence
   seg.segment_text("Kumain siya ng pagkain.")
   # ['um', 'kain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']
