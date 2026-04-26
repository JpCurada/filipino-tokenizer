Researcher Guide
================

This guide explains the linguistic theory behind the tokenizer, the algorithmic
design decisions, how to evaluate it, and how to extend it to other Philippine languages.

Filipino morphology primer
--------------------------

Tagalog is an **agglutinative language** — complex words are formed by attaching
affixes to a root.  Unlike English, where affixes attach only at word edges,
Tagalog also uses **infixes** that are inserted *inside* the root.

Affix types
~~~~~~~~~~~

+------------+----------+-------------------+-------------------------------------------+
| Type       | Example  | Segmentation      | Meaning                                   |
+============+==========+===================+===========================================+
| Prefix     | pagkain  | pag + kain        | "food" (pag- nominalises)                 |
+------------+----------+-------------------+-------------------------------------------+
| Infix -um- | kumain   | k + um + ain      | "ate" (-um- marks actor focus, past)      |
+------------+----------+-------------------+-------------------------------------------+
| Infix -in- | kinain   | k + in + ain      | "was eaten" (-in- marks object focus)     |
+------------+----------+-------------------+-------------------------------------------+
| Suffix     | kainan   | kain + an         | "dining place" (-an locative)             |
+------------+----------+-------------------+-------------------------------------------+
| Circumfix  | pagkainan| pag + kain + an   | "dining hall" (pag- -an together)         |
+------------+----------+-------------------+-------------------------------------------+

Infixes are particularly important for tokenisation.  The surface form ``kumain``
does not begin with the root ``kain``; instead the root's first consonant ``k``
comes first, then the infix ``um``, then the rest of the root ``ain``.
A character-level tokenizer sees ``k``, ``u``, ``m``, ``a``, ``i``, ``n``
with no concept that ``kain`` is the meaningful unit.

Nasal assimilation
~~~~~~~~~~~~~~~~~~

The prefixes ``pang-`` and ``mang-`` undergo **nasal assimilation** when the root
begins with certain consonants, which changes both the prefix surface form and
drops the root's initial consonant:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Root initial consonant
     - Surface prefix
     - Example
   * - b, p
     - pam- / mam-
     - pamili (pang + bili)
   * - d, t, s
     - pan- / man-
     - panulat (pang + sulat)
   * - k, g
     - pang- / mang-
     - pangkain (pang + kain)
   * - vowel, h, l, m, n, w, y
     - pang- / mang-
     - pangasiwa (pang + asiwa)

The ``TagalogPhonology`` class handles forward (apply) and reverse (strip)
direction for these rules.

----

The Constrained BPE algorithm
------------------------------

Background
~~~~~~~~~~

Standard `Byte Pair Encoding <https://arxiv.org/abs/1508.07909>`_ (BPE) learns
subword units by repeatedly merging the most frequent adjacent pair of symbols in a
corpus.  Applied naively to Filipino, it produces merges that cross morpheme
boundaries — e.g., merging ``n`` and ``g`` in ``pagkain`` to create ``ng``
regardless of whether ``n`` and ``g`` belong to different morphemes.

The CBPE constraint
~~~~~~~~~~~~~~~~~~~

This library implements **Constrained BPE** (CBPE), following the approach of
Tacorda et al. (2024).  The constraint is simple:

  **No merge may combine two symbols that are separated by a morpheme boundary marker.**

The boundary marker is ``▁`` (U+2581, LOWER ONE EIGHTH BLOCK), the same character
used by SentencePiece.

Pipeline
~~~~~~~~

::

   Raw text
      │
      ▼
   Pre-tokenize         Split on whitespace and punctuation
      │
      ▼
   Morphological        TagalogSegmenter identifies morphemes;
   Segmentation         TagalogTokenizer inserts ▁ into the surface text
      │                 at morpheme boundaries
      ▼
   Surface-annotated    e.g. "pag▁kain" for pagkain
   tokens               e.g. "k▁um▁ain" for kumain (infix)
      │
      ▼
   CBPE Training        BPE pair-counting skips any pair that
   (or Encoding)        spans a ▁ boundary

The critical detail for **infix forms**: the segmenter returns ``['um', 'kain']``
for ``kumain``, but these morphemes do not concatenate to give the surface word.
The ``_surface_annotate`` method maps them back to the surface text with boundary
markers: ``k▁um▁ain``.  This means:

- ``k`` and ``um`` cannot be merged (``▁`` between them)
- ``um`` and ``ain`` cannot be merged (``▁`` between them)
- ``k`` and ``a`` cannot be merged (not adjacent in the token sequence — ``um`` is between them)

The root fragment ``kain`` is therefore *split* in infix words, which is unavoidable
given the phonological reality of Tagalog infixation.  For prefix/suffix forms
(``pag▁kain``) the root ``kain`` appears intact and receives consistent token IDs.

Heap-based incremental BPE
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MorphAwareBPE`` training loop uses an optimised incremental algorithm:

1. **Doubly-linked list** — each unique word sequence is represented as a linked
   list of ``Node`` objects, enabling O(1) local edits when a merge is applied.
2. **Max-heap with lazy deletion** — the most frequent pair is found in O(log n)
   time.  Stale heap entries (whose count has decreased since they were pushed) are
   skipped at pop time.
3. **Position index** — ``pair_positions[pair]`` is a set of nodes where the pair
   starts, enabling targeted updates instead of a full corpus rescan.

This brings training complexity from O(N²) (naïve BPE) down to O(N log V) where
N is corpus size and V is vocabulary size.

----

Morpheme segmentation passes
-----------------------------

The ``TagalogSegmenter`` runs five passes in order, returning the first successful
segmentation:

.. list-table::
   :header-rows: 1
   :widths: 5 20 50 25

   * - Pass
     - Name
     - Logic
     - Example
   * - 0
     - Frozen-form guard
     - If the whole word is a root *and* stripping a prefix yields another root
       with an identical dictionary definition, return the word unsegmented.
     - *pangalan* → ``['pangalan']`` (not ``pang + alan``)
   * - 1
     - Circumfix
     - Try all (prefix, suffix) circumfix pairs longest-first. Accept if the
       core is ≥ 4 chars, is a root, and is not a redundant duplicate.
     - *pagkainan* → ``['pag', 'kain', 'an']``
   * - 2
     - Prefix (recursive)
     - Strip the longest matching prefix. Recurse on the remainder (up to depth 3)
       to handle stacked prefixes. Try infix detection on the remainder before
       accepting a bare root.
     - *pinakamahusay* → ``['pinaka', 'ma', 'husay']``
   * - 3
     - Infix
     - Check whether inserting ``-um-`` or ``-in-`` after the first consonant
       gives a valid root (≥ 4 chars, in dictionary).
     - *kumain* → ``['um', 'kain']``
   * - 4
     - Suffix
     - Strip suffix variants (including h-insertion: ``-an``/``-han``,
       ``-in``/``-hin``). Accept if root is ≥ 4 chars and in dictionary.
     - *kainan* → ``['kain', 'an']``
   * - 5
     - Fallback
     - Return ``[word]`` unsegmented.
     - *computer* → ``['computer']``

Root validation
~~~~~~~~~~~~~~~

Every candidate root is checked against ``tagalog_roots.json`` (~28,000 entries).
The minimum root length is 4 characters (``_MIN_ROOT = 4``), which eliminates
spurious matches against short dictionary fragments like ``gka`` or ``nda`` that
appear in the roots file as inflected-form artefacts.

Redundancy check
~~~~~~~~~~~~~~~~

The ``_is_redundant(word, root)`` method compares the dictionary *definitions* of
the whole word and the candidate root.  If they are identical, the segmentation is
rejected — this catches duplicate entries like:

- ``pangalan`` — definition: "name; reputation; repute; denomination"
- ``alan`` — definition: "name; reputation; repute; denomination"

Without this check, the segmenter would produce ``['pang', 'alan']`` for a word
that is itself a frozen lexical entry.

----

Evaluation methodology
-----------------------

Morpheme boundary accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary metric used in the demo notebooks is **morpheme boundary F1**:

- **Gold standard**: manually verified morpheme segmentations for ~200 words
  across prefixed, infixed, suffixed, circumfixed, stacked-prefix, and
  unsegmentable categories.
- **Predicted boundaries**: token split positions output by the tokenizer.
- **F1**: harmonic mean of precision (fraction of predicted boundaries that are
  gold) and recall (fraction of gold boundaries that are predicted).

.. code-block:: python

   def get_boundaries(segments):
       boundaries = set()
       pos = 0
       for s in segments[:-1]:
           pos += len(s)
           boundaries.add(pos)
       return boundaries

   def compute_f1(gold, pred):
       hits = len(gold & pred)
       prec = hits / len(pred) if pred else 0.0
       rec  = hits / len(gold) if gold else 0.0
       return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

Fertility
~~~~~~~~~

**Fertility** = tokens per word.  Lower fertility means the tokenizer is
compressing Filipino words into more meaningful units:

.. code-block:: python

   tokens_per_word = len(tok.encode(sentence)) / len(sentence.split())

Root consistency
~~~~~~~~~~~~~~~~

For a given root (e.g., *kain*), encode the root alone, then check whether those
exact IDs appear as a contiguous subsequence in the encoding of each inflected form.
For prefix/suffix forms this will always hold; for infix forms it will not
(the root is split around the infix), which is expected.

----

Extending to a new language
----------------------------

The library is designed for multiple Philippine languages.  All affix data is stored
in four shared JSON files (``data/prefix_table.json`` etc.) filtered by a
``"language"`` field.  Adding a new language requires:

1. **Add affix entries** to the JSON tables:

   .. code-block:: json

      {
        "mag-": [
          {"language": "Tagalog", "function": "...", "etymology": "..."},
          {"language": "Bisaya",  "function": "...", "etymology": "..."}
        ]
      }

2. **Add a root file** — ``data/<language>_roots.json``, same schema as
   ``tagalog_roots.json``:

   .. code-block:: json

      [
        {"word": "kaon", "definition": "to eat", "language": "Bisaya",
         "part_of_speech": "v", "link": ""}
      ]

3. **Create an affixes class**:

   .. code-block:: python

      # src/<language>/affixes.py
      from filipino_tokenizer.base import BaseAffixes

      class BisayaAffixes(BaseAffixes):
          def __init__(self):
              super().__init__(language="Bisaya")

4. **Create a roots class**:

   .. code-block:: python

      from filipino_tokenizer.base import BaseRoots

      class BisayaRoots(BaseRoots):
          def __init__(self):
              super().__init__(language="Bisaya", filename="bisaya_roots.json")

5. **Create a phonology class** — subclass or replace ``TagalogPhonology`` with
   language-specific rules (Bisaya has different nasal assimilation patterns).

6. **Create a segmenter** — subclass ``BaseSegmenter``, implementing the same
   pass structure with language-appropriate adjustments.

7. **Create a tokenizer** — wire the segmenter into ``MorphAwareBPE``, following
   ``TagalogTokenizer`` as a template.

----

References
----------

- **Tacorda, Livelo, Ong, and Cheng (2024).**
  Constraining Byte Pair Encoding (CBPE) to improve morphological segmentation
  for Filipino tokenizers.
  *The CBPE approach this library implements.*

- **Sennrich, Haddow, and Birch (2016).**
  Neural Machine Translation of Rare Words with Subword Units.
  *ACL 2016.*  `arXiv:1508.07909 <https://arxiv.org/abs/1508.07909>`_
  *The original BPE paper.*

- **Cruz, J.P. and Cheng, C. (2022).**
  Improving Large-scale Language Models and Resources for Filipino.
  *Source of Filipino NLP benchmarks referenced in evaluation.*

- **Miranda, L.J. (2023).**
  calamanCy: A Tagalog Natural Language Processing Toolkit.
  *SpaCy-based Tagalog pipeline that informed morphological analysis design.*
