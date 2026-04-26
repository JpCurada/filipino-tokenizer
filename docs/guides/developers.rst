Developer Guide
===============

This guide covers everything you need to integrate Filipino Tokenizer into an
application or ML pipeline.

Corpus preparation
------------------

The tokenizer reads a plain UTF-8 text file — one sentence per line.

.. code-block:: text

   Kumain siya ng pagkain sa hapagkainan.
   Ang mga bata ay masayang naglalaro sa labas.
   Maganda ang panahon ngayon.

**Size guidelines**

+-------------------+--------------------------------------------+
| Corpus size       | Recommended use                            |
+===================+============================================+
| < 10k sentences   | Prototyping / demos only                   |
+-------------------+--------------------------------------------+
| 10k – 100k        | Small-scale experiments                    |
+-------------------+--------------------------------------------+
| 100k – 1M         | Production NLP tasks                       |
+-------------------+--------------------------------------------+
| > 1M              | Large language model pre-training          |
+-------------------+--------------------------------------------+

A good starting corpus for Tagalog is the `WikiText-TL-39 dataset
<https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks>`_.

**Writing a corpus programmatically**

.. code-block:: python

   import tempfile, os

   sentences = [
       "Kumain siya ng pagkain.",
       "Maganda ang panahon ngayon.",
   ]

   with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                    encoding="utf-8") as f:
       f.write("\n".join(sentences))
       corpus_path = f.name

----

Training
--------

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.train(corpus_path, vocab_size=32000)

   print(f"Vocab size  : {len(tok.bpe.vocab)}")
   print(f"Merge rules : {len(tok.bpe.merges)}")

**Choosing vocab_size**

``vocab_size`` sets an upper bound on the BPE vocabulary.  A larger vocabulary
means longer, more-complete tokens (lower fertility) but increases model embedding
table size.

+-------------+-------------------------------------------------------+
| vocab_size  | Typical use case                                      |
+=============+=======================================================+
| 500 – 2000  | Small experiments, unit tests                         |
+-------------+-------------------------------------------------------+
| 8000        | Lightweight production tokenizer                      |
+-------------+-------------------------------------------------------+
| 32000       | Standard for transformer language models              |
+-------------+-------------------------------------------------------+
| 64000+      | Very large corpora / multilingual settings            |
+-------------+-------------------------------------------------------+

.. note::

   If the corpus is too small to generate ``vocab_size`` unique merges, training
   stops early.  Check ``len(tok.bpe.merges)`` after training to see the actual count.

----

Encoding
--------

Single sentence
~~~~~~~~~~~~~~~

.. code-block:: python

   ids = tok.encode("Kumain siya ng pagkain.")
   # [79, 99, 115, 99, 133, 4, 154, 4, 100, 4, 125, 99, 145, 18]

- Input is lowercased automatically.
- Returns ``list[int]``.
- Unknown characters fall back to the ``<unk>`` token (ID 1).

Inspecting tokens as strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   tokens = tok.tokenize("Kumain siya ng pagkain.")
   # ['k', 'um', 'ain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']

Batch encoding
~~~~~~~~~~~~~~

There is no built-in batch method.  Use a list comprehension:

.. code-block:: python

   sentences = ["Kumain siya.", "Maganda ang panahon."]
   batch_ids = [tok.encode(s) for s in sentences]

For large batches, use ``concurrent.futures`` to parallelise:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor() as pool:
       batch_ids = list(pool.map(tok.encode, sentences))

----

Decoding
--------

.. code-block:: python

   text = tok.decode(ids)
   # 'kumain siya ng pagkain.'

- Special tokens (``<pad>``, ``<unk>``, ``<s>``, ``</s>``) are silently dropped.
- Boundary markers (``▁``) are removed.
- Output is always lowercase (encoding lowercases input).

----

Saving and loading
------------------

.. code-block:: python

   # Save
   tok.save("my_tokenizer/")
   # Creates my_tokenizer/vocab.json and my_tokenizer/merges.txt

   # Load
   from filipino_tokenizer.tagalog import TagalogTokenizer
   tok2 = TagalogTokenizer()
   tok2.load("my_tokenizer/")

The saved files are human-readable plain text — you can inspect or version-control them.

----

Using the segmenter independently
----------------------------------

The morphological segmenter can be used without the BPE layer:

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogSegmenter

   seg = TagalogSegmenter()

   seg.segment("kumain")          # ['um', 'kain']
   seg.segment("pagkain")         # ['pag', 'kain']
   seg.segment("pinakamahusay")   # ['pinaka', 'ma', 'husay']
   seg.segment("pangalan")        # ['pangalan']  ← frozen form, not decomposed
   seg.segment("computer")        # ['computer']  ← loan word, not decomposed

   # Segment a full sentence (splits on whitespace/punctuation first)
   seg.segment_text("Kumain siya ng pagkain.")
   # ['um', 'kain', ' ', 'siya', ' ', 'ng', ' ', 'pag', 'kain', '.']

This is useful for:

- Feature extraction for non-BPE models
- Linguistic analysis and corpus statistics
- Preprocessing for other NLP tools

----

Integrating with PyTorch
------------------------

Filipino Tokenizer produces plain Python lists, which convert directly to tensors:

.. code-block:: python

   import torch
   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.load("my_tokenizer/")

   def collate(sentences, pad_id=0):
       encoded = [tok.encode(s) for s in sentences]
       max_len = max(len(e) for e in encoded)
       padded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
       return torch.tensor(padded, dtype=torch.long)

   batch = collate(["Kumain siya.", "Maganda ang panahon ngayon."])
   # tensor of shape (2, max_seq_len)

----

Integrating with HuggingFace ``datasets``
------------------------------------------

.. code-block:: python

   from datasets import Dataset
   from filipino_tokenizer.tagalog import TagalogTokenizer

   tok = TagalogTokenizer()
   tok.load("my_tokenizer/")

   raw = Dataset.from_dict({"text": ["Kumain siya.", "Maganda ang panahon."]})

   def tokenize_fn(batch):
       return {"input_ids": [tok.encode(t) for t in batch["text"]]}

   tokenized = raw.map(tokenize_fn, batched=True)

----

Special token IDs
-----------------

+----------+----+---------------------------------------------+
| Token    | ID | Meaning                                     |
+==========+====+=============================================+
| ``<pad>``| 0  | Padding (for fixed-length batches)          |
+----------+----+---------------------------------------------+
| ``<unk>``| 1  | Unknown character fallback                  |
+----------+----+---------------------------------------------+
| ``<s>``  | 2  | Beginning of sequence                       |
+----------+----+---------------------------------------------+
| ``</s>`` | 3  | End of sequence                             |
+----------+----+---------------------------------------------+

These are always at IDs 0–3 regardless of corpus.  Add them manually if your model
expects them:

.. code-block:: python

   BOS, EOS = 2, 3
   ids = [BOS] + tok.encode(sentence) + [EOS]

----

HuggingFace Transformers integration
--------------------------------------

``TagalogHFTokenizer`` implements the ``PreTrainedTokenizer`` interface so it works
directly with any HuggingFace-compatible training framework.

.. code-block:: bash

   pip install filipino-tokenizer[hf]

Loading a trained tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from filipino_tokenizer.tagalog import TagalogHFTokenizer

   tok = TagalogHFTokenizer(
       vocab_file="my_tokenizer/vocab.json",
       merges_file="my_tokenizer/merges.txt",
   )

   print(tok.vocab_size)      # 32000
   print(tok.bos_token)       # '<s>'
   print(tok.pad_token_id)    # 0

Batch encoding
~~~~~~~~~~~~~~

.. code-block:: python

   sentences = [
       "Kumain siya ng pagkain.",
       "Nagtatrabaho ang tatay sa opisina araw-araw.",
   ]
   encoding = tok(sentences, padding=True, truncation=True, return_tensors="pt")
   # encoding["input_ids"]       — shape (2, seq_len)
   # encoding["attention_mask"]  — shape (2, seq_len)

Save and reload in HuggingFace format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   tok.save_pretrained("hf_tokenizer/")
   # Creates: vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json

   tok2 = TagalogHFTokenizer.from_pretrained("hf_tokenizer/")

Building a dataset for causal LM training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import Dataset

   class FilipinoTextDataset(Dataset):
       def __init__(self, texts, tokenizer, max_length=512):
           self.encodings = tokenizer(
               texts,
               max_length=max_length,
               padding="max_length",
               truncation=True,
               return_tensors="pt",
           )

       def __len__(self):
           return self.encodings["input_ids"].shape[0]

       def __getitem__(self, idx):
           item = {k: v[idx] for k, v in self.encodings.items()}
           item["labels"] = item["input_ids"].clone()
           return item

Setting up a model
~~~~~~~~~~~~~~~~~~

The only tokenizer-specific value a model needs is ``vocab_size``:

.. code-block:: python

   from transformers import GPT2Config, GPT2LMHeadModel

   config = GPT2Config(
       vocab_size=tok.vocab_size,
       pad_token_id=tok.pad_token_id,
       bos_token_id=tok.bos_token_id,
       eos_token_id=tok.eos_token_id,
   )
   model = GPT2LMHeadModel(config)

The same pattern works for any architecture (``LlamaForCausalLM``,
``BertForMaskedLM``, ``T5ForConditionalGeneration``, etc.) — only the
config class changes.

----

Running tests
-------------

.. code-block:: bash

   python -m unittest discover tests -v

All 49 tests should pass.  Individual test files:

.. code-block:: bash

   python -m unittest tests.test_affixes -v
   python -m unittest tests.test_segmenter -v
   python -m unittest tests.test_tokenizer -v
