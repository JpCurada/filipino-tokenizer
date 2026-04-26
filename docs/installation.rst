Installation
============

Requirements
------------

- Python 3.10 or later
- No external runtime dependencies — the core library uses only the standard library
  (``json``, ``os``, ``re``, ``collections``, ``heapq``)

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

The core library has no optional dependencies.  The demo notebooks in ``demo/``
use ``plotly``, ``tiktoken``, and ``sentencepiece`` for comparisons — install
those separately if you want to run the notebooks:

.. code-block:: bash

   pip install plotly tiktoken sentencepiece jupyter
