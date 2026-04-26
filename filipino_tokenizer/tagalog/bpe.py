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
        # Inference cache
        self._encode_cache: dict[str, list[int]] = {}

    # ================================================================== #
    #  Training                                                            #
    # ================================================================== #

    def train(self, corpus: list[str], vocab_size: int = 32_000) -> None:
        """
        Train the BPE vocabulary from *corpus* (list of pre-segmented strings
        with ``BOUNDARY`` markers between morphemes).

        Uses an optimized incremental algorithm:
        1. Doubly-linked list for each word.
        2. Max-heap for finding the most frequent pair.
        3. Lazy deletion for stale heap entries.
        """
        self._encode_cache.clear()
        import heapq

        word_freqs: dict[tuple[str, ...], int] = Counter()
        for entry in corpus:
            symbols = tuple(entry)
            if symbols:
                word_freqs[symbols] += 1

        self._init_base_vocab(word_freqs)

        class Node:
            __slots__ = ['token', 'prev', 'next', 'freq', 'deleted']
            def __init__(self, token: str, freq: int):
                self.token = token
                self.prev: 'Node' | None = None
                self.next: 'Node' | None = None
                self.freq = freq
                self.deleted = False

        pair_counts: dict[tuple[str, str], int] = {}
        pair_positions: dict[tuple[str, str], set[Node]] = {}

        # 1. Build doubly-linked list for each unique word sequence
        for symbols, freq in word_freqs.items():
            if len(symbols) < 2:
                continue
                
            prev_node = None
            for sym in symbols:
                node = Node(sym, freq)
                if prev_node:
                    prev_node.next = node
                    node.prev = prev_node
                    
                    pair = (prev_node.token, node.token)
                    if BOUNDARY not in pair[0] and BOUNDARY not in pair[1]:
                        pair_counts[pair] = pair_counts.get(pair, 0) + freq
                        if pair not in pair_positions:
                            pair_positions[pair] = set()
                        pair_positions[pair].add(prev_node)
                        
                prev_node = node

        # 2. Initialize max-heap
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        # 3. Iteratively merge
        import sys
        num_merges = vocab_size - len(self.vocab)
        report_every = max(1, num_merges // 20)  # report every ~5%
        for merge_i in range(num_merges):
            best_pair = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                count = -neg_count
                if pair_counts.get(pair, 0) == count:
                    best_pair = pair
                    break

            if not best_pair:
                print(
                    f"  100.0%  {merge_i:,}/{num_merges:,} merges (no more pairs)          ",
                    file=sys.stderr,
                )
                break

            if (merge_i + 1) % report_every == 0 or merge_i == num_merges - 1:
                pct = (merge_i + 1) / num_merges * 100
                print(
                    f"  {pct:5.1f}%  {merge_i + 1:,}/{num_merges:,} merges",
                    end="\r", file=sys.stderr, flush=True,
                )
            if merge_i == num_merges - 1:
                print(file=sys.stderr)  # newline after final update
                
            a, b = best_pair
            merged_token = a + b
            
            # Register new token
            new_id = len(self.vocab)
            self.vocab[merged_token] = new_id
            self.id_to_token[new_id] = merged_token
            self.merges.append(best_pair)
            
            # Process all occurrences of this pair
            nodes_to_process = list(pair_positions.get(best_pair, []))
            
            for node in nodes_to_process:
                if node.deleted:
                    continue
                if not node.next or node.next.deleted:
                    continue
                if node.token != a or node.next.token != b:
                    continue
                    
                prev_node = node.prev
                next_node = node.next.next
                
                # Decrement old pairs
                if prev_node:
                    old_pair1 = (prev_node.token, node.token)
                    if BOUNDARY not in old_pair1[0] and BOUNDARY not in old_pair1[1]:
                        pair_counts[old_pair1] -= node.freq
                        pair_positions[old_pair1].discard(prev_node)
                        if pair_counts[old_pair1] > 0:
                            heapq.heappush(heap, (-pair_counts[old_pair1], old_pair1))
                            
                pair_counts[best_pair] -= node.freq
                pair_positions[best_pair].discard(node)
                if pair_counts[best_pair] > 0:
                    heapq.heappush(heap, (-pair_counts[best_pair], best_pair))
                    
                if next_node:
                    old_pair2 = (node.next.token, next_node.token)
                    if BOUNDARY not in old_pair2[0] and BOUNDARY not in old_pair2[1]:
                        pair_counts[old_pair2] -= node.freq
                        pair_positions[old_pair2].discard(node.next)
                        if pair_counts[old_pair2] > 0:
                            heapq.heappush(heap, (-pair_counts[old_pair2], old_pair2))
                            
                # Merge nodes
                node.token = merged_token
                node.next.deleted = True
                node.next = next_node
                if next_node:
                    next_node.prev = node
                    
                # Increment new pairs
                if prev_node:
                    new_pair1 = (prev_node.token, node.token)
                    if BOUNDARY not in new_pair1[0] and BOUNDARY not in new_pair1[1]:
                        pair_counts[new_pair1] = pair_counts.get(new_pair1, 0) + node.freq
                        if new_pair1 not in pair_positions:
                            pair_positions[new_pair1] = set()
                        pair_positions[new_pair1].add(prev_node)
                        heapq.heappush(heap, (-pair_counts[new_pair1], new_pair1))
                        
                if next_node:
                    new_pair2 = (node.token, next_node.token)
                    if BOUNDARY not in new_pair2[0] and BOUNDARY not in new_pair2[1]:
                        pair_counts[new_pair2] = pair_counts.get(new_pair2, 0) + node.freq
                        if new_pair2 not in pair_positions:
                            pair_positions[new_pair2] = set()
                        pair_positions[new_pair2].add(node)
                        heapq.heappush(heap, (-pair_counts[new_pair2], new_pair2))

    # ------------------------------------------------------------------ #
    #  Internal training helpers                                           #
    # ------------------------------------------------------------------ #

    def _init_base_vocab(self, word_freqs: dict[tuple[str, ...], int]) -> None:
        """Populate vocab with specials + all printable characters + corpus chars."""
        self.vocab.clear()
        self.id_to_token.clear()
        self.merges.clear()
        self._encode_cache.clear()

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
            
        if text in self._encode_cache:
            return list(self._encode_cache[text])

        # Start with characters
        symbols = list(text)
        
        # Apply merges in learned order
        # Note: Since train() never generated merges with BOUNDARY,
        # _apply_merges will naturally stop at BOUNDARY markers.
        symbols = self._apply_merges(symbols)
        
        # Convert to IDs
        all_ids: list[int] = []
        for sym in symbols:
            if sym in self.vocab:
                all_ids.append(self.vocab[sym])
            else:
                # Character fallback for unseen symbols
                for char in sym:
                    all_ids.append(self.vocab.get(char, self.vocab.get(self.UNK, 1)))

        self._encode_cache[text] = list(all_ids)
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
        self._encode_cache.clear()
        
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
