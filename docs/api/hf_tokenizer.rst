TagalogHFTokenizer
==================

.. autoclass:: filipino_tokenizer.tagalog.hf_tokenizer.TagalogHFTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

----

Overview
--------

``TagalogHFTokenizer`` wraps :class:`~filipino_tokenizer.tagalog.tokenizer.TagalogTokenizer`
behind the HuggingFace ``PreTrainedTokenizer`` interface.  It requires
``transformers>=4.30``:

.. code-block:: bash

   pip install filipino-tokenizer[hf]

----

Method reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Signature
     - Description
   * - ``__call__``
     - ``(text, padding, truncation, return_tensors, ...)``
     - Standard HF tokenizer call — returns ``input_ids``, ``attention_mask``.
   * - ``encode``
     - ``(text) → list[int]``
     - Encode a single string to token IDs.
   * - ``decode``
     - ``(ids) → str``
     - Decode token IDs back to text.
   * - ``save_pretrained``
     - ``(directory)``
     - Save in HF format (adds ``tokenizer_config.json``).
   * - ``from_pretrained``
     - ``(directory_or_repo)``
     - Load from a local directory or HuggingFace Hub repo.

----

Attributes
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``vocab_size``
     - Total vocabulary size (32,000 for the pretrained model).
   * - ``bos_token`` / ``bos_token_id``
     - ``"<s>"`` / ``2``
   * - ``eos_token`` / ``eos_token_id``
     - ``"</s>"`` / ``3``
   * - ``pad_token`` / ``pad_token_id``
     - ``"<pad>"`` / ``0``
   * - ``unk_token`` / ``unk_token_id``
     - ``"<unk>"`` / ``1``

----

Examples
--------

Load and encode
~~~~~~~~~~~~~~~

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogHFTokenizer

   tok = TagalogHFTokenizer(
       vocab_file="my_tokenizer/vocab.json",
       merges_file="my_tokenizer/merges.txt",
   )

   encoding = tok("Kumain siya ng pagkain.", return_tensors="pt")
   # {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}

Batch with padding
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sentences = [
       "Kumain siya ng pagkain.",
       "Nagtatrabaho ang tatay sa opisina araw-araw.",
   ]
   encoding = tok(sentences, padding=True, truncation=True,
                  max_length=128, return_tensors="pt")

Save and reload
~~~~~~~~~~~~~~~

.. code-block:: python

   tok.save_pretrained("hf_tokenizer/")
   tok2 = TagalogHFTokenizer.from_pretrained("hf_tokenizer/")

Use with a language model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import GPT2Config, GPT2LMHeadModel

   model = GPT2LMHeadModel(GPT2Config(
       vocab_size=tok.vocab_size,
       pad_token_id=tok.pad_token_id,
       bos_token_id=tok.bos_token_id,
       eos_token_id=tok.eos_token_id,
   ))
