"""
Morphology-Aware Byte Pair Encoding (BPE) for Filipino languages.

Key constraint (Constrained BPE / CBPE):
    During training, merges are NEVER applied across morpheme boundary
    markers ("▁").  This preserves linguistic morpheme boundaries in the
    final subword vocabulary so that morphologically meaningful units are
    kept intact.

Implementation notes:
    - No external dependencies (HuggingFace, sentencepiece, etc.).
    - Uses only standard library: json, os, re, collections.
    - Boundary marker "▁" (U+2581 LOWER ONE EIGHTH BLOCK) is the
      conventional SentencePiece / Unigram separator.
"""

import json
import os
import re
from collections import Counter


# Morpheme boundary marker inserted between segmented morphemes
BOUNDARY = "▁"


class MorphAwareBPE:
    """
    Byte-Pair Encoding tokenizer with a morpheme-boundary constraint.

    During ``train()``, the algorithm counts bigram frequencies across the
    corpus but **skips** any pair that spans a ``BOUNDARY`` marker.  This
    guarantees that learned merges never glue together parts of different
    morphemes.

    Vocabulary layout:
        id 0  →  <pad>
        id 1  →  <unk>
        id 2  →  <s>
        id 3  →  </s>
        id 4 … 259  →  individual bytes (all 256 byte values)
        id 260 …  →  learned merge tokens
    """

    # Special tokens ---------------------------------------------------- #
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    SPECIALS = [PAD, UNK, BOS, EOS]

    def __init__(self):
        # token_str → id
        self.vocab: dict[str, int] = {}
        # id → token_str
        self.id_to_token: dict[int, str] = {}
        # ordered list of (token_a, token_b) merge pairs
        self.merges: list[tuple[str, str]] = []

    # ================================================================== #
    #  Training                                                            #
    # ================================================================== #

    def train(self, corpus: list[str], vocab_size: int = 32_000) -> None:
        """
        Train the BPE vocabulary from *corpus* (list of pre-segmented strings
        with ``BOUNDARY`` markers between morphemes).

        Parameters
        ----------
        corpus : list[str]
            Each element is a segmented word / token sequence, e.g.
            ``"pag▁kain"`` meaning prefix *pag* + root *kain*.
        vocab_size : int
            Target vocabulary size (including specials + byte tokens).
        """

        # 1. Build initial per-word symbol sequences.
        # Each corpus entry is split into characters, but BOUNDARY markers
        # stay as single symbols so we can detect (and skip) cross-boundary
        # merges.
        word_freqs: dict[tuple[str, ...], int] = Counter()
        for entry in corpus:
            symbols = tuple(entry)           # each char is a symbol
            if symbols:
                word_freqs[symbols] += 1

        # 2. Initialise base vocab: specials + byte-level characters.
        self._init_base_vocab(word_freqs)

        # 3. Iteratively merge the most frequent valid pair.
        num_merges = vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts = self._count_pairs(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            merged = best_pair[0] + best_pair[1]

            # Register the new token
            new_id = len(self.vocab)
            self.vocab[merged] = new_id
            self.id_to_token[new_id] = merged
            self.merges.append(best_pair)

            # Apply the merge to every word in the frequency table
            word_freqs = self._apply_merge(word_freqs, best_pair, merged)

    # ------------------------------------------------------------------ #
    #  Internal training helpers                                           #
    # ------------------------------------------------------------------ #

    def _init_base_vocab(self, word_freqs: dict[tuple[str, ...], int]) -> None:
        """Populate vocab with specials + all printable characters + corpus chars."""
        self.vocab.clear()
        self.id_to_token.clear()
        self.merges.clear()

        # Specials
        for tok in self.SPECIALS:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok

        # Base character set: all printable ASCII (32-126) + boundary marker
        # + any additional characters found in the corpus (e.g. accented chars).
        # This ensures that punctuation and other common characters can always
        # be encoded, even if they weren't in the training corpus.
        chars: set[str] = set()
        for code in range(32, 127):
            chars.add(chr(code))
        chars.add(BOUNDARY)
        # Add all characters actually seen in the corpus
        for symbols in word_freqs:
            chars.update(symbols)
        for ch in sorted(chars):
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.id_to_token[idx] = ch

    def _count_pairs(
        self, word_freqs: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, str], int]:
        """
        Count adjacent symbol pairs, **skipping** any pair where either
        symbol is or contains the morpheme boundary marker.
        """
        counts: dict[tuple[str, str], int] = {}
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                a, b = symbols[i], symbols[i + 1]
                # CBPE constraint: never merge across a boundary
                if BOUNDARY in a or BOUNDARY in b:
                    continue
                pair = (a, b)
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    @staticmethod
    def _apply_merge(
        word_freqs: dict[tuple[str, ...], int],
        pair: tuple[str, str],
        merged: str,
    ) -> dict[tuple[str, ...], int]:
        """Replace every occurrence of *pair* in the symbol sequences."""
        new_freqs: dict[tuple[str, ...], int] = {}
        a, b = pair
        for symbols, freq in word_freqs.items():
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == a
                    and symbols[i + 1] == b
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_freqs[tuple(new_symbols)] = (
                new_freqs.get(tuple(new_symbols), 0) + freq
            )
        return new_freqs

    # ================================================================== #
    #  Encoding                                                            #
    # ================================================================== #

    def encode(self, text: str) -> list[int]:
        """
        Encode *text* into a list of token IDs.

        The text is first split into individual characters, then merge rules
        are applied in order.  Unknown characters map to ``<unk>``.
        """
        if not text:
            return []

        # Split into segments separated by boundary markers
        segments = text.split(BOUNDARY)
        all_ids: list[int] = []

        for seg_idx, segment in enumerate(segments):
            if not segment:
                continue
            # Start with characters
            symbols = list(segment)
            # Apply merges in learned order
            symbols = self._apply_merges(symbols)
            # Convert to IDs
            for sym in symbols:
                all_ids.append(
                    self.vocab.get(sym, self.vocab.get(self.UNK, 1))
                )

        return all_ids

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Apply all learned merges to *symbols* in order."""
        for a, b in self.merges:
            merged = a + b
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == a
                    and symbols[i + 1] == b
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    # ================================================================== #
    #  Decoding                                                            #
    # ================================================================== #

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Special tokens (<pad>, <unk>, <s>, </s>) are silently dropped.
        Boundary markers are removed so the output reads naturally.
        """
        tokens: list[str] = []
        for token_id in ids:
            tok = self.id_to_token.get(token_id, self.UNK)
            if tok in self.SPECIALS:
                continue
            tokens.append(tok)
        text = "".join(tokens)
        # Remove any remaining boundary markers
        text = text.replace(BOUNDARY, "")
        return text

    # ================================================================== #
    #  Serialisation                                                       #
    # ================================================================== #

    def save(self, directory: str) -> None:
        """
        Persist the tokenizer to *directory* as two files:

        - ``vocab.json``  — maps token string → integer id
        - ``merges.txt``  — one merge per line, ``token_a<TAB>token_b``
        """
        os.makedirs(directory, exist_ok=True)

        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        merges_path = os.path.join(directory, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a}\t{b}\n")

    def load(self, directory: str) -> None:
        """Load a previously saved tokenizer from *directory*."""
        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Rebuild reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        merges_path = os.path.join(directory, "merges.txt")
        self.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))
