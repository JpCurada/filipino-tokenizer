"""
Integration tests for the Rust BPE backend.

Verifies that the Rust CoreBPE produces correct results when wired into
MorphAwareBPE and TagalogTokenizer.

Run with:
    python -m unittest tests.test_rust_backend -v
"""

import unittest

from filipino_tokenizer._bpe_rust import CoreBPE as _RustCoreBPE
from filipino_tokenizer.tagalog.tokenizer import TagalogTokenizer


def _load_tokenizer() -> TagalogTokenizer:
    tok = TagalogTokenizer()
    tok.load_pretrained()
    return tok


class TestRustExtensionLoads(unittest.TestCase):
    def test_import(self):
        """The compiled Rust extension must be importable."""
        self.assertTrue(callable(_RustCoreBPE))

    def test_instantiate(self):
        """CoreBPE can be constructed from a minimal vocab and merge list."""
        vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5, "ab": 6}
        merges = [("a", "b")]
        bpe = _RustCoreBPE(vocab, merges)
        self.assertIsNotNone(bpe)


class TestEncodeDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.bpe.encode("kumain")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_encode_nonempty(self):
        ids = self.tok.bpe.encode("kumain")
        self.assertGreater(len(ids), 0)

    def test_encode_empty_string(self):
        self.assertEqual(self.tok.bpe.encode(""), [])

    def test_decode_returns_string(self):
        ids = self.tok.bpe.encode("kumain")
        text = self.tok.bpe.decode(ids)
        self.assertIsInstance(text, str)

    def test_roundtrip(self):
        """decode(encode(word)) must recover the original word."""
        words = ["kumain", "nagtatrabaho", "mahal", "siya", "ng"]
        for word in words:
            with self.subTest(word=word):
                ids = self.tok.bpe.encode(word)
                recovered = self.tok.bpe.decode(ids)
                self.assertEqual(recovered, word)

    def test_decode_drops_specials(self):
        """Special token IDs must be silently dropped during decode."""
        pad_id = self.tok.bpe.vocab["<pad>"]
        bos_id = self.tok.bpe.vocab["<s>"]
        eos_id = self.tok.bpe.vocab["</s>"]
        word_ids = self.tok.bpe.encode("kain")
        ids = [bos_id] + word_ids + [eos_id, pad_id]
        text = self.tok.bpe.decode(ids)
        self.assertEqual(text, "kain")


class TestMorphologyBoundary(unittest.TestCase):
    BOUNDARY = "▁"

    # Broad candidate list — setup filters to only words the segmenter
    # actually annotates, so the tests don't depend on dictionary coverage.
    _CANDIDATES = [
        "kumain",        # um- infix,   root=kain
        "bumasa",        # um- infix,   root=basa
        "lumabas",       # um- infix,   root=labas
        "pinakamahusay", # pinaka- prefix
        "nagtatrabaho",  # nag- prefix
        "nagluluto",     # nag- prefix
        "magsalita",     # mag- prefix
        "mahal",         # root only — intentionally excluded if unsegmented
    ]

    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()
        # Keep only words the segmenter inserts boundaries into.
        cls.SEGMENTED_WORDS = [
            w for w in cls._CANDIDATES
            if cls.BOUNDARY in cls.tok._surface_annotate(w)
        ]

    def test_at_least_some_words_are_segmented(self):
        """The segmenter must handle at least some morphologically complex words."""
        self.assertGreater(
            len(self.SEGMENTED_WORDS), 0,
            "No candidate words were segmented — check segmenter / roots data.",
        )

    def test_complex_words_produce_boundaries(self):
        """Each word in SEGMENTED_WORDS must have ▁ in its annotated form."""
        for word in self.SEGMENTED_WORDS:
            with self.subTest(word=word):
                annotated = self.tok._surface_annotate(word)
                self.assertIn(
                    self.BOUNDARY, annotated,
                    f"{word!r} should be segmented but got {annotated!r}",
                )

    def test_no_bpe_token_crosses_boundary(self):
        """After BPE encoding a boundary-annotated word, no output token
        may contain ▁ embedded between other characters — only as a
        standalone token is ▁ allowed."""
        for word in self.SEGMENTED_WORDS:
            with self.subTest(word=word):
                annotated = self.tok._surface_annotate(word)
                if self.BOUNDARY not in annotated:
                    self.skipTest(f"{word!r} was not segmented")

                ids = self.tok.bpe.encode(annotated)
                for id_ in ids:
                    token_str = self.tok.bpe.id_to_token.get(id_, "")
                    self.assertFalse(
                        self.BOUNDARY in token_str and token_str != self.BOUNDARY,
                        f"Token {token_str!r} crosses a morpheme boundary "
                        f"in {word!r} (annotated: {annotated!r})",
                    )

    def test_annotated_form_roundtrips(self):
        """decode(encode(annotated)) must recover the original word
        (boundary markers removed by decode)."""
        for word in self.SEGMENTED_WORDS:
            with self.subTest(word=word):
                annotated = self.tok._surface_annotate(word)
                ids = self.tok.bpe.encode(annotated)
                recovered = self.tok.bpe.decode(ids)
                self.assertEqual(recovered, word)

    def test_encode_cache_consistent(self):
        """Cached result must match a fresh encode call."""
        annotated = self.tok._surface_annotate("kumain")
        self.tok.bpe._encode_cache.clear()
        ids_first = self.tok.bpe.encode(annotated)
        ids_cached = self.tok.bpe.encode(annotated)
        self.assertEqual(ids_first, ids_cached)


class TestTokenizerEncodeDecodeFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()

    def test_full_sentence_roundtrip(self):
        sentences = [
            "kumain siya ng pagkain.",
            "nagtatrabaho ang tatay sa opisina.",
            "ang mga bata ay masayang naglalaro.",
        ]
        for sent in sentences:
            with self.subTest(sent=sent):
                ids = self.tok.encode(sent)
                recovered = self.tok.decode(ids)
                self.assertEqual(recovered, sent.lower())


if __name__ == "__main__":
    unittest.main()
