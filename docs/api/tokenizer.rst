TagalogTokenizer
================

.. autoclass:: filipino_tokenizer.tagalog.tokenizer.TagalogTokenizer
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
     - ``(corpus_path, vocab_size=32000)``
     - Train BPE from a plain-text corpus file.
   * - ``encode``
     - ``(text) → list[int]``
     - Encode text to token IDs.
   * - ``decode``
     - ``(ids) → str``
     - Decode token IDs back to text.
   * - ``tokenize``
     - ``(text) → list[str]``
     - Return subword strings instead of IDs (for inspection).
   * - ``load_pretrained``
     - ``()``
     - Load the bundled 32k model shipped with the package. No path needed.
   * - ``save``
     - ``(directory)``
     - Write ``vocab.json`` and ``merges.txt`` to *directory*.
   * - ``load``
     - ``(directory)``
     - Load a previously saved tokenizer.

----

Attributes
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``tok.bpe``
     - The underlying :class:`~filipino_tokenizer.tagalog.bpe.MorphAwareBPE` instance.
       Access ``tok.bpe.vocab`` (dict), ``tok.bpe.merges`` (list of tuples),
       ``tok.bpe.id_to_token`` (dict).
   * - ``tok.segmenter``
     - The underlying :class:`~filipino_tokenizer.tagalog.segmenter.TagalogSegmenter`
       instance. Use ``tok.segmenter.segment(word)`` independently.

----

Examples
--------

Load the bundled pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No download or path required — the 32k model is shipped with the package:

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.load_pretrained()
   ids = tok.encode("Kumain siya ng pagkain.")

Train on your own corpus
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.train("corpus.txt", vocab_size=32000)

Encode / decode round-trip
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ids = tok.encode("Kumain siya ng pagkain.")
   assert tok.decode(ids) == "kumain siya ng pagkain."

Inspect tokens
~~~~~~~~~~~~~~

.. code-block:: python

   tok.tokenize("Pinakamahusay ang ginawa niya.")
   # ['pinaka', '▁', 'ma', '▁', 'husay', ' ', 'ang', ' ', ...]

Save and reload
~~~~~~~~~~~~~~~

.. code-block:: python

   tok.save("my_tokenizer/")

   tok2 = TagalogTokenizer()
   tok2.load("my_tokenizer/")
   assert tok.encode("test") == tok2.encode("test")
