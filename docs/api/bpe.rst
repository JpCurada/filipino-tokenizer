MorphAwareBPE
=============

.. autoclass:: filipino_tokenizer.tagalog.bpe.MorphAwareBPE
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
   * - ``train``
     - ``(corpus, vocab_size=32000)``
     - Train BPE from a list of pre-annotated strings (with ``▁`` markers).
   * - ``encode``
     - ``(text) → list[int]``
     - Encode a boundary-annotated string to token IDs.
   * - ``decode``
     - ``(ids) → str``
     - Decode token IDs back to a string (boundary markers removed).
   * - ``save``
     - ``(directory)``
     - Write ``vocab.json`` and ``merges.txt``.
   * - ``load``
     - ``(directory)``
     - Load a previously saved BPE model.

----

Vocabulary layout
-----------------

+----------+----+-------------------------------------------------+
| Token    | ID | Notes                                           |
+==========+====+=================================================+
| ``<pad>``| 0  | Always present                                  |
+----------+----+-------------------------------------------------+
| ``<unk>``| 1  | Unknown character fallback                      |
+----------+----+-------------------------------------------------+
| ``<s>``  | 2  | Beginning of sequence                           |
+----------+----+-------------------------------------------------+
| ``</s>`` | 3  | End of sequence                                 |
+----------+----+-------------------------------------------------+
| chars    | 4+ | All printable ASCII (32–126) + ``▁`` + corpus   |
|          |    | characters, sorted, allocated in order          |
+----------+----+-------------------------------------------------+
| merges   | …  | Learned BPE merge tokens, in training order     |
+----------+----+-------------------------------------------------+

----

The CBPE constraint
-------------------

During ``train()``, the algorithm counts bigram frequencies across the corpus but
**skips any pair that contains a** ``▁`` **boundary marker**.  Concretely, in
``_init_pair_counts()``:

.. code-block:: python

   if BOUNDARY not in pair[0] and BOUNDARY not in pair[1]:
       pair_counts[pair] += freq

This guarantees that no learned merge rule ever combines tokens from different
morphemes.

----

Saving and loading
------------------

``save(directory)`` writes two files:

- ``vocab.json`` — JSON object mapping token string → integer ID.
- ``merges.txt`` — one merge per line, ``token_a<TAB>token_b``.

Both files are UTF-8 and human-readable.

.. code-block:: python

   from filipino_tokenizer.tagalog.bpe import MorphAwareBPE, BOUNDARY

   bpe = MorphAwareBPE()
   bpe.train([f"pag{BOUNDARY}kain", f"ma{BOUNDARY}ganda"] * 10, vocab_size=100)
   bpe.save("bpe_model/")

   bpe2 = MorphAwareBPE()
   bpe2.load("bpe_model/")
   assert bpe.encode(f"pag{BOUNDARY}kain") == bpe2.encode(f"pag{BOUNDARY}kain")

----

Constants
---------

.. autodata:: filipino_tokenizer.tagalog.bpe.BOUNDARY
   :annotation: = "▁"

The boundary marker (U+2581 LOWER ONE EIGHTH BLOCK) inserted between morphemes
in surface-annotated text.  Identical to the SentencePiece word-boundary character.
