# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.13 with a local `.venv` (created via `python -m venv .venv`)
- Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
- No `pyproject.toml`, `setup.py`, or `requirements.txt` exists yet — the project is in early development

## Running code

```bash
# Run from the repo root so that `src` imports resolve correctly
python -c "from src.tagalog.affixes import TagalogAffixes; a = TagalogAffixes(); print(a.get_prefixes())"

# Run the example script
python examples/training_tagalog_tokenizer.py

# Open the demo notebook
jupyter notebook demo/demo_tagalog_tokenizer.ipynb
```

## Architecture

The library provides morphological tokenization for Filipino languages (Tagalog and Cebuano), built around affix-stripping using linguistically curated data files.

### Data layer (`data/`)

Four JSON affix tables drive everything:
- `prefix_table.json`, `suffix_table.json`, `infix_table.json`, `circumfix_table.json` — keyed by affix string (e.g. `"di-"`, `"-in-"`), each entry has `language`, `function`, `etymology`, `pronunciation`, `derived_terms`
- `tagalog_roots.json`, `bisaya_roots.json` — root word lists

Each affix entry supports multiple languages in an array; `BaseAffixes._load()` filters by the `language` field and takes the first matching entry.

### Source layer (`src/`)

- `src/base.py` — `BaseAffixes`: loads and filters all affix tables by language; exposes `get_prefixes()`, `get_suffixes()`, `get_infixes()`, `get_circumfixes()`, all sorted longest-first to enable greedy matching
- `src/tagalog/affixes.py` — `TagalogAffixes(BaseAffixes)`: filters for `language="Tagalog"`
- `src/cebuano/affixes.py` — `CebuanoAffixes(BaseAffixes)`: filters for `language="Cebuano"`
- `src/tagalog/tokenizer.py`, `src/tagalog/segmenter.py`, `src/tagalog/phonology.py` — stubs (empty files, not yet implemented)

### Import convention

All imports use `from src.<module> import ...` with the repo root on `sys.path`. There is no installed package — run scripts from the project root.

### Adding a new language

1. Add entries to the JSON tables with the new `language` value
2. Create `src/<language>/affixes.py` subclassing `BaseAffixes` with `super().__init__(language="<Language>")`
3. Add root words JSON to `data/` and load it in `BaseAffixes` or the language subclass
