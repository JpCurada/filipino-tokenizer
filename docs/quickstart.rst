Quick Start
===========

This page gets you from zero to a working tokenizer in under two minutes.

1. Prepare a corpus
-------------------

The tokenizer trains on a plain UTF-8 text file with **one sentence per line**.

.. code-block:: text

   Kumain siya ng pagkain sa hapagkainan.
   Maganda ang panahon ngayon kaya lumabas kami.
   Nagluluto ang nanay ng masarap na adobo para sa pamilya.

Save this as ``corpus.txt``.  For production use, download the
`Wikitext-TL-39 <https://huggingface.co/datasets/linkanjarad/Wikitext-TL39>`_
corpus (~1.5M sentences) with the included script:

.. code-block:: bash

   pip install datasets
   python scripts/download_corpus.py

2. Train
--------

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.train("corpus.txt", vocab_size=32000)

``vocab_size`` is the target BPE vocabulary size.  The actual vocabulary
will be smaller if the corpus does not contain enough distinct character pairs.

3. Encode and decode
--------------------

.. code-block:: python

   ids = tok.encode("Kumain siya ng pagkain.")
   # [79, 99, 115, ...]

   text = tok.decode(ids)
   # 'kumain siya ng pagkain.'

``encode()`` lowercases input and returns a ``list[int]``.
``decode()`` removes boundary markers and reconstructs the original text.

4. Inspect tokens
-----------------

.. code-block:: python

   tokens = tok.tokenize("Kumain siya ng pagkain.")
   # ['k', 'um', 'ain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']

``tokenize()`` returns strings instead of IDs — useful for debugging and
understanding what the tokenizer is doing.

5. Save and reload
------------------

.. code-block:: python

   tok.save("my_tokenizer/")

   tok2 = TagalogTokenizer()
   tok2.load("my_tokenizer/")

This writes two files:

- ``my_tokenizer/vocab.json`` — token-to-ID mapping
- ``my_tokenizer/merges.txt`` — learned BPE merge rules

6. HuggingFace integration
--------------------------

``TagalogHFTokenizer`` wraps the tokenizer behind the ``PreTrainedTokenizer``
interface for use with ``Trainer``, TRL, Axolotl, and any other HF pipeline.

.. code-block:: bash

   pip install filipino-tokenizer[hf]

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogHFTokenizer

   tok = TagalogHFTokenizer(
       vocab_file="my_tokenizer/vocab.json",
       merges_file="my_tokenizer/merges.txt",
   )

   # Standard HuggingFace call
   encoding = tok("Kumain siya ng pagkain.", return_tensors="pt")

   # Save / reload in HF format
   tok.save_pretrained("hf_tokenizer/")
   tok2 = TagalogHFTokenizer.from_pretrained("hf_tokenizer/")

See :doc:`api/hf_tokenizer` for the full API reference.

What's next?
------------

- **Developers** — see :doc:`guides/developers` for corpus preparation,
  batch encoding, and integration with ML frameworks.
- **Researchers** — see :doc:`guides/researchers` for the morphological
  segmentation algorithm, the CBPE constraint, and evaluation methodology.
- **API details** — see :doc:`api/tokenizer`, :doc:`api/segmenter`,
  :doc:`api/bpe`, :doc:`api/hf_tokenizer`.
