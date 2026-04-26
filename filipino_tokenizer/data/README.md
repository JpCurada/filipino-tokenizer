# Filipino Tokenizer Data Resources

This directory contains the linguistic resources (JSON tables) required to drive the `TagalogSegmenter` and related classes.

These resources are designed to be language-agnostic at the file level (supporting multiple Philippine languages). Subclasses like `TagalogAffixes` load these files and filter them by the `language` key.

## Files

- **`prefix_table.json`**: Contains definitions of prefixes, including their potential recursive nature (stacked prefixes).
- **`suffix_table.json`**: Contains definitions of suffixes, including phonological variants (e.g., `-an` vs `-han`).
- **`infix_table.json`**: Contains infixes (e.g., `-um-`, `-in-`).
- **`circumfix_table.json`**: Contains circumfixes (paired prefixes and suffixes).
- **`tagalog_roots.json`**: A comprehensive list of ~30,000 root words specific to Tagalog, used to validate segmentation candidates.
- **`bisaya_roots.json`**: A foundational list of root words for Bisaya/Cebuano (WIP).

*Note: The `eval/` directory (if present locally) contains large training and evaluation corpora derived from WikiText, but these are ignored in version control due to file size limits.*
