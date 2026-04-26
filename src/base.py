import json
import os
import re


class BaseAffixes:
    def __init__(self, language: str):
        self.language = language
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        self.prefixes = self._load("prefix_table.json")
        self.suffixes = self._load("suffix_table.json")
        self.infixes = self._load("infix_table.json")
        self.circumfixes = self._load_circumfixes("circumfix_table.json")

    def _load(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        filtered = {}
        for affix_key, entries in raw.items():
            for entry in entries:
                if entry.get("language") == self.language:
                    clean = affix_key.strip("-").strip()
                    filtered[clean] = {
                        "raw": affix_key,
                        "function": entry.get("function", ""),
                        "etymology": entry.get("etymology", ""),
                        "pronunciation": entry.get("pronunciation", ""),
                        "derived_terms": entry.get("derived_terms", []),
                    }
                    break
        return filtered

    def _load_circumfixes(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        filtered = {}
        for affix_key, entries in raw.items():
            for entry in entries:
                if entry.get("language") == self.language:
                    parts = affix_key.split()
                    prefix = parts[0].strip("-")
                    suffix = parts[-1].strip("-")
                    filtered[affix_key] = {
                        "prefix": prefix,
                        "suffix": suffix,
                        "raw": affix_key,
                        "function": entry.get("function", ""),
                        "etymology": entry.get("etymology", ""),
                        "derived_terms": entry.get("derived_terms", []),
                    }
                    break
        return filtered

    def get_prefixes(self):
        """Return prefix strings sorted longest first."""
        return sorted(self.prefixes.keys(), key=len, reverse=True)

    def get_suffixes(self):
        """Return suffix strings sorted longest first."""
        return sorted(self.suffixes.keys(), key=len, reverse=True)

    def get_infixes(self):
        """Return infix strings."""
        return list(self.infixes.keys())

    def get_circumfixes(self):
        """Return circumfix (prefix, suffix) tuples sorted by prefix length longest first."""
        pairs = [
            (v["prefix"], v["suffix"])
            for v in self.circumfixes.values()
        ]
        return sorted(pairs, key=lambda x: len(x[0]), reverse=True)


class BaseRoots:
    def __init__(self, language: str, filename: str):
        self.language = language
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self._roots = {}
        self._load(filename)

    def _load(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for entry in raw:
            if entry.get("language") == self.language:
                word = entry.get("word", "").lower().strip()
                if word and word not in self._roots:
                    self._roots[word] = {
                        "definition": entry.get("definition", ""),
                        "part_of_speech": entry.get("part_of_speech"),
                        "link": entry.get("link", ""),
                    }

    def is_root(self, word: str) -> bool:
        return word.lower() in self._roots

    def get_root_info(self, word: str):
        return self._roots.get(word.lower())

    def get_all_roots(self) -> set:
        return set(self._roots.keys())


class BaseSegmenter:
    def segment(self, word: str) -> list:
        raise NotImplementedError

    def segment_text(self, text: str) -> list:
        """Tokenize text on whitespace/punctuation, then segment each word."""
        result = []
        for token in re.split(r'(\s+|[^\w])', text):
            if not token:
                continue
            if re.match(r'^\w+$', token):
                result.extend(self.segment(token.lower()))
            else:
                result.append(token)
        return result


class BaseTokenizer:
    def encode(self, text: str) -> list:
        raise NotImplementedError

    def decode(self, tokens: list) -> str:
        raise NotImplementedError
