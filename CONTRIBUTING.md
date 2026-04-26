# Contributing to Filipino Tokenizer

First off, thank you for considering contributing to the Filipino Tokenizer! It's people like you that make open-source tools for Philippine languages possible.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Adding a New Language](#adding-a-new-language)
4. [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you are expected to uphold a welcoming, inclusive, and respectful environment. Be kind to others, and constructive in your feedback.

## Development Setup

1. **Fork and Clone**
   Fork the repo and clone it locally:
   ```bash
   git clone https://github.com/<your-username>/filipino-tokenizer.git
   cd filipino-tokenizer
   ```

2. **Virtual Environment**
   Set up your environment and install the package in editable mode with development dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate # Linux/macOS
   
   pip install -e .[dev]
   ```

3. **Running Tests**
   We use the built-in `unittest` framework. Before submitting a PR, ensure all tests pass:
   ```bash
   python -m unittest discover tests -v
   ```

## Adding a New Language

This tokenizer is designed to be extensible to other Philippine languages (Cebuano/Bisaya, Ilokano, Hiligaynon, etc.). To add a new language:

1. **Update Data Files**: Add the affixes for your language to the JSON files in `filipino_tokenizer/data/` (e.g., `prefix_table.json`). Be sure to specify the `"language"` key.
2. **Add Roots**: Create a root dictionary file (e.g., `filipino_tokenizer/data/cebuano_roots.json`).
3. **Subclass Core Components**: Create a new folder (e.g., `filipino_tokenizer/cebuano/`) and implement the `Affixes`, `Roots`, and `Segmenter` classes by inheriting from the `Base` classes in `filipino_tokenizer/base.py`.
4. **Implement Phonology**: Ensure any language-specific phonological rules (like Tagalog's nasal assimilation) are correctly implemented in your segmenter.

## Pull Request Process

1. Create a descriptive branch name (`feat/cebuano-support`, `fix/cache-bug`).
2. Keep your PRs focused. If you are adding a language, don't mix it with core framework refactoring.
3. Write unit tests for your changes. If you add a new phonological rule, add a test for it.
4. Fill out the Pull Request template completely.
5. Wait for a maintainer to review your code. We will provide constructive feedback!
