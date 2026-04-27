Installation
============

Requirements
------------

- Python 3.10 or later
- No external runtime dependencies

  Pre-built wheels are published for Linux, macOS, and Windows on Python 3.10–3.13.
  ``pip install`` downloads the right binary — no compiler or Rust toolchain needed.

  .. note::

     Installing from source (e.g. cloning the repo and running ``pip install -e .``)
     requires a Rust toolchain. See `rustup.rs <https://rustup.rs>`_ to install one.

From PyPI
---------

.. code-block:: bash

   pip install filipino-tokenizer

From source
-----------

.. code-block:: bash

   git clone https://github.com/JpCurada/filipino-tokenizer.git
   cd filipino-tokenizer
   pip install -e .

The ``-e`` flag installs in *editable* mode, so changes to the source are reflected
immediately without reinstalling.

Verify the installation
-----------------------

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer, TagalogSegmenter

   seg = TagalogSegmenter()
   print(seg.segment("kumain"))   # ['um', 'kain']

Optional dependencies
---------------------

HuggingFace integration
~~~~~~~~~~~~~~~~~~~~~~~

To use ``TagalogHFTokenizer`` with ``transformers``-based training pipelines:

.. code-block:: bash

   pip install filipino-tokenizer[hf]

This installs ``transformers>=4.30``.

Demo notebooks
~~~~~~~~~~~~~~

The notebooks in ``demo/`` use additional packages for comparisons and visualisations:

.. code-block:: bash

   pip install plotly tiktoken sentencepiece jupyter

Corpus download
~~~~~~~~~~~~~~~

To download the Wikitext-TL-39 training corpus:

.. code-block:: bash

   pip install datasets
   python scripts/download_corpus.py
