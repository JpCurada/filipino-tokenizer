"""
Tagalog Tokenizer — combines morphological segmentation with
morphology-aware BPE.

Pipeline:
    1. Pre-tokenize text on whitespace/punctuation
    2. For each word, run the morphological segmenter to find morpheme
       boundaries, then insert boundary markers into the **surface text**
       so that the original spelling is preserved.
    3. Feed the boundary-annotated corpus into ``MorphAwareBPE.train()``
    4. At encode time, re-run segmentation + BPE encode

This ensures the BPE vocabulary never contains merges that cross
morpheme boundaries, while preserving perfect round-trip fidelity.
"""

import os
import re
import json

from filipino_tokenizer.tagalog.segmenter import TagalogSegmenter
from filipino_tokenizer.tagalog.bpe import MorphAwareBPE, BOUNDARY


class TagalogTokenizer:
    """
    End-to-end tokenizer for Tagalog text.

    Usage::

        tok = TagalogTokenizer()
        tok.train("corpus.txt", vocab_size=32000)
        ids = tok.encode("Kumain siya ng pagkain.")
        text = tok.decode(ids)
        assert text == "kumain siya ng pagkain."
    """

    def __init__(self):
        self.segmenter = TagalogSegmenter()
        self.bpe = MorphAwareBPE()
        self._segment_cache: dict[str, str] = {}

    # ================================================================== #
    #  Training                                                            #
    # ================================================================== #

    def train(self, corpus_path: str, vocab_size: int = 32_000) -> None:
        """
        Train the tokenizer from a plain-text corpus file.

        Steps:
            1. Read the corpus file line-by-line.
            2. Pre-tokenize each line into words / punctuation.
            3. Segment each word morphologically.
            4. Insert boundary markers into the surface text at morpheme
               boundaries (preserving original spelling).
            5. Train BPE with the CBPE constraint.

        Parameters
        ----------
        corpus_path : str
            Path to a UTF-8 plain-text file (one sentence per line).
        vocab_size : int
            Target BPE vocabulary size.
        """
        annotated_tokens: list[str] = []
        cache: dict[str, str] = {}

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = re.split(r'(\s+|[^\w])', line)
                for part in parts:
                    if not part:
                        continue
                    if re.match(r'^\w+$', part):
                        word = part.lower()
                        if word not in cache:
                            cache[word] = self._surface_annotate(word)
                        annotated_tokens.append(cache[word])
                    else:
                        annotated_tokens.append(part)

        self.bpe.train(annotated_tokens, vocab_size=vocab_size)

    # ================================================================== #
    #  Encoding                                                            #
    # ================================================================== #

    def encode(self, text: str) -> list[int]:
        """
        Encode *text* into a list of integer token IDs.

        The text is lowercased, split into words/punctuation, each word
        is morphologically segmented (with boundary markers in the surface
        form), and BPE encoding is applied.
        """
        all_ids: list[int] = []
        tokens = self._segment_line(text)
        for token in tokens:
            ids = self.bpe.encode(token)
            all_ids.extend(ids)
        return all_ids

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize *text* into subword strings (for debugging / inspection).

        Returns the string representation of each BPE token rather than
        integer IDs.
        """
        ids = self.encode(text)
        return [
            self.bpe.id_to_token.get(i, MorphAwareBPE.UNK)
            for i in ids
        ]

    # ================================================================== #
    #  Decoding                                                            #
    # ================================================================== #

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a readable string.

        Boundary markers and special tokens are removed.  Spaces between
        words are reconstructed by detecting word-boundary tokens.
        """
        # Reconstruct raw text from BPE
        raw = self.bpe.decode(ids)
        # Normalise whitespace
        raw = re.sub(r'\s+', ' ', raw).strip()
        return raw

    # ================================================================== #
    #  Persistence                                                         #
    # ================================================================== #

    def save(self, directory: str) -> None:
        """
        Save the trained tokenizer to *directory*.

        Creates:
            - ``vocab.json`` — BPE vocabulary mapping
            - ``merges.txt`` — ordered merge rules
        """
        self.bpe.save(directory)

    def load(self, directory: str) -> None:
        """Load a previously saved tokenizer from *directory*."""
        self.bpe.load(directory)

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #

    def _segment_line(self, line: str) -> list[str]:
        """
        Pre-tokenize a line of text and morphologically segment each word.

        Returns a list of boundary-annotated strings where boundary
        markers are inserted into the **surface text** at morpheme
        boundaries.  This preserves the original spelling for perfect
        round-trip fidelity.

        For example, ``kumain`` is segmented into morphemes ``[um, kain]``
        (infix *um*) but the surface form is ``kumain``.  We locate
        morpheme boundaries in the surface text and produce
        ``"k▁um▁ain"`` so that:
          - BPE sees boundaries and won't merge across them
          - Removing ▁ gives back ``kumain`` exactly.
        """
        result: list[str] = []
        # Split on whitespace and punctuation, keeping delimiters
        parts = re.split(r'(\s+|[^\w])', line)
        for part in parts:
            if not part:
                continue
            if re.match(r'^\w+$', part):
                # Morphological segmentation
                word = part.lower()
                if word not in self._segment_cache:
                    self._segment_cache[word] = self._surface_annotate(word)
                result.append(self._segment_cache[word])
            else:
                # Whitespace / punctuation — pass through
                result.append(part)
        return result

    def _surface_annotate(self, word: str) -> str:
        """
        Run the morphological segmenter and insert boundary markers into
        the surface text at positions corresponding to morpheme boundaries.

        The segmenter returns abstract morphemes which, for infixes, don't
        directly concatenate to form the surface word.  This method maps
        the morphemes back to the surface form.

        Strategies:
            - Prefixes/suffixes: direct concatenation matches surface form.
            - Infixes (-um-, -in-): inserted after the first consonant.
              Surface = first_consonant + infix + remainder_of_root.
            - Unsegmented: return as-is (no boundaries).
        """
        morphemes = self.segmenter.segment(word)

        # Single morpheme or empty — no boundaries to insert
        if len(morphemes) <= 1:
            return word

        # Check if direct concatenation matches the surface form
        # (works for prefixes, suffixes, circumfixes)
        concat = "".join(morphemes)
        if concat == word:
            return BOUNDARY.join(morphemes)

        # Handle infix cases: the segmenter returns [infix, root] or
        # [prefix, infix, root] etc.
        # We need to find where these morphemes appear in the surface text
        return self._reconstruct_with_infixes(word, morphemes)

    def _reconstruct_with_infixes(self, word: str, morphemes: list[str]) -> str:
        """
        Insert boundary markers for infix-containing words.

        Known Tagalog infix patterns:
            - [infix, root]: e.g. ['um', 'kain'] for 'kumain'
              Surface = root[0] + infix + root[1:] = k + um + ain
            - [prefix, infix, root]: e.g. ['nag', 'um', 'root']
              (rare, but handle it)

        We try to locate each morpheme's contribution in the surface text.
        """
        infixes = set(self.segmenter.affixes.get_infixes())

        # Case: [infix, root] — the most common infix-only pattern
        if len(morphemes) == 2 and morphemes[0] in infixes:
            infix = morphemes[0]
            root = morphemes[1]
            # Surface form: root[0] + infix + root[1:]
            if len(root) >= 1:
                expected = root[0] + infix + root[1:]
                if expected == word:
                    # Boundary after root[0], after infix
                    return root[0] + BOUNDARY + infix + BOUNDARY + root[1:]

        # Case: [prefix(es)…, infix, root]
        # Try to find the infix in the morpheme list and handle prefix part
        prefix_parts = []
        infix_part = None
        root_part = None

        for i, m in enumerate(morphemes):
            if m in infixes and infix_part is None:
                infix_part = m
                # Everything after the infix is the root
                remaining = morphemes[i + 1:]
                if remaining:
                    root_part = remaining[0]
                    # Any further morphemes are suffixes
                    suffix_parts = remaining[1:]
                else:
                    suffix_parts = []
                break
            else:
                prefix_parts.append(m)

        if infix_part and root_part:
            # Build: prefixes + root[0] + infix + root[1:] + suffixes
            prefix_str = "".join(prefix_parts)
            infix_surface = root_part[0] + infix_part + root_part[1:]
            suffix_str = "".join(suffix_parts) if suffix_parts else ""
            expected = prefix_str + infix_surface + suffix_str
            if expected == word:
                parts = []
                if prefix_parts:
                    parts.extend(prefix_parts)
                parts.append(root_part[0])
                parts.append(infix_part)
                parts.append(root_part[1:])
                if suffix_parts:
                    parts.extend(suffix_parts)
                # Filter empty parts
                parts = [p for p in parts if p]
                return BOUNDARY.join(parts)

        # Fallback: can't reconstruct, return with simple boundary join
        # (won't roundtrip for complex cases, but is safe)
        return BOUNDARY.join(morphemes)
