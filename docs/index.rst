Filipino Tokenizer
==================

**Morphology-aware BPE tokenization for Philippine languages.**

Filipino words are built by stacking prefixes, infixes, suffixes, and circumfixes
onto a root.  A generic tokenizer trained on English treats this morphology as noise
and splits words at arbitrary character positions.  Filipino Tokenizer fixes that:
it uses a rule-based morphological segmenter to identify morpheme boundaries *before*
running BPE, so the learned subword units are always linguistically meaningful.

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.train("corpus.txt", vocab_size=32000)

   tok.tokenize("Kumain siya ng pagkain.")
   # ['k', 'um', 'ain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']

The root *kain* (eat) appears as a single token in both *kumain* and *pagkain*,
even though the surface forms look very different.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/developers
   guides/researchers

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/tokenizer
   api/segmenter
   api/bpe
   api/hf_tokenizer

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
