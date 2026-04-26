# Filipino Tokenizer - Project Context

## What This Is
A morphology-aware BPE tokenizer for Philippine languages.
First target: Tagalog. Future: Bisaya, Ilokano, others.

The key innovation: combining a structured Filipino affix database 
with BPE subword tokenization. The segmenter decomposes Filipino 
words into morphemes, then BPE operates with a constraint to never 
merge across morpheme boundaries.

## Project Structure
- `data/` — shared JSON files for all languages (DO NOT modify these)
- `src/base.py` — shared base classes (BaseAffixes, BaseSegmenter, BaseTokenizer)
- `src/tagalog/` — Tagalog-specific implementation
- `src/bisaya/` — Bisaya-specific implementation (future)
- `tests/` — test files
- `demo/` — Jupyter notebooks
- `examples/` — example scripts

## Data Schemas

### Affix JSONs (prefix_table.json, suffix_table.json, infix_table.json)
```json
{
  "affix-key": [
    {"language": "Tagalog", "function": "...", "etymology": "...", "pronunciation": "...", "derived_terms": []},
    {"language": "Cebuano", "function": "...", ...}
  ]
}
```
Keys are affix strings with dashes (e.g. "mag-", "-in-", "-an").
Each key maps to a list of entries, one per language. Filter by "language" field.

### Circumfix JSON (circumfix_table.json)
Same structure but keys have two parts: "mag- -an", "ka- -han", "pag- -in".
Split on whitespace to get prefix and suffix parts.

### Roots JSON (tagalog_roots.json, bisaya_roots.json)
```json
[
  {"word": "kain", "definition": "...", "language": "Cebuano", "part_of_speech": "n", "link": "..."}
]
```
Array of objects. Filter by "language" field.
Note: tagalog_roots.json may have language set to "Tagalog" — check actual value.

## Architecture Rules
- All languages share the same data/ JSON files, filtered by language key
- BaseAffixes in base.py handles loading and filtering — language modules just subclass
- Prefixes are always sorted longest-first for longest-match-first segmentation
- Segmenter returns list[str] always — unsegmentable words return [word]
- BPE is implemented from scratch — no HuggingFace tokenizers dependency
- Only standard library: json, os, re, collections. No external deps for core.

## Morphological Segmenter Logic (segmenter.py)
Multi-pass approach in this order:
1. Circumfix detection (check prefix+suffix combos first)
2. Prefix stripping (longest match first, can loop for stacked prefixes)
3. Suffix stripping (apply phonology rules for -han/-hin variants)
4. Infix detection (-um- and -in- after first consonant)
5. Reduplication detection (CV partial reduplication)
6. Root validation (check against root dictionary, backtrack if invalid)
Fallback: return word unsegmented

## Phonology Rules (phonology.py)
Nasal assimilation for pang-/mang-:
- + b/p → pam-/mam- (drop first letter of root)
- + d/t → pan-/man- (drop first letter of root)  
- + k/g → pang-/mang- (may drop first letter)
- + s → pan-/man- (drop first letter)
- + vowel/h/l/m/n/w/y → pang-/mang- (root stays)

Suffix phonology:
- root ends in vowel + -an → -han
- root ends in vowel + -in → -hin

## Build Order
1. src/base.py
2. src/tagalog/roots.py
3. src/tagalog/affixes.py
4. src/tagalog/phonology.py
5. tests/test_affixes.py
6. src/tagalog/segmenter.py
7. tests/test_segmenter.py
8. src/tagalog/bpe.py
9. src/tagalog/tokenizer.py
10. tests/test_tokenizer.py

## Common Pitfalls
- "pangalan" (name) should NOT be segmented as pang- + alan. Root validation catches this.
- Loan words (computer, internet) should not be segmented. If no valid root found, return whole.
- The infix -in- and prefix in- are different. Context matters.
- Some affixes exist in both Tagalog and Cebuano with different functions — always filter by language.