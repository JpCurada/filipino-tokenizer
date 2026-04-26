Contributing
============

Thank you for considering contributing to the Filipino Tokenizer!
The full contributing guide lives in the repository root:
`CONTRIBUTING.md <https://github.com/JpCurada/filipino-tokenizer/blob/main/CONTRIBUTING.md>`_.

Development Setup
-----------------

1. Fork and clone the repo:

   .. code-block:: bash

      git clone https://github.com/<your-username>/filipino-tokenizer.git
      cd filipino-tokenizer

2. Create a virtual environment and install in editable mode:

   .. code-block:: bash

      python -m venv .venv
      .venv\Scripts\activate      # Windows
      # source .venv/bin/activate # Linux/macOS

      pip install -e .[dev]

3. Run the test suite before submitting a PR:

   .. code-block:: bash

      python -m unittest discover tests -v

Adding a New Language
---------------------

The tokenizer is designed to be extensible to other Philippine languages
(Cebuano/Bisaya, Ilokano, Hiligaynon, etc.):

1. Add affix entries to the JSON files in ``filipino_tokenizer/data/`` with the
   appropriate ``"language"`` key.
2. Create a root dictionary (e.g., ``filipino_tokenizer/data/cebuano_roots.json``).
3. Subclass the core components in a new folder
   (e.g., ``filipino_tokenizer/cebuano/``):
   ``Affixes``, ``Roots``, ``Segmenter``, and ``Tokenizer``.
4. Implement language-specific phonological rules in your segmenter.

Pull Request Process
--------------------

1. Create a descriptive branch name (``feat/cebuano-support``, ``fix/cache-bug``).
2. Keep PRs focused — don't mix language additions with core refactoring.
3. Write unit tests for your changes.
4. Fill out the pull request template completely.
5. Wait for a maintainer review.
