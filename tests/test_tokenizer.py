"""
Tests for the full Tagalog tokenizer pipeline.

Covers:
    1. Round-trip: encode → decode returns original text
    2. Morpheme consistency: shared root produces consistent tokens
    3. Efficiency: Filipino sentences produce fewer tokens than char-level baseline
    4. End-to-end: train on a small corpus, verify encode/decode works
"""

import os
import tempfile
import unittest

from filipino_tokenizer.tagalog.tokenizer import TagalogTokenizer
from filipino_tokenizer.tagalog.bpe import MorphAwareBPE, BOUNDARY


# --------------------------------------------------------------------- #
#  Small Filipino training corpus                                        #
# --------------------------------------------------------------------- #

SMALL_CORPUS = """\
Kumain siya ng pagkain sa hapagkainan.
Ang mga bata ay masayang naglalaro sa labas.
Maganda ang panahon ngayon kaya lumabas kami.
Bumili ako ng mga prutas sa palengke kahapon.
Nagluluto ang nanay ng masarap na adobo para sa pamilya.
Pumunta kami sa simbahan tuwing Linggo.
Naglalakad sila sa tabing-dagat tuwing hapon.
Nagbabasa ang mga estudyante ng libro sa silid-aklatan.
Kumanta ang mga bata sa programa ng paaralan.
Umuulan kaya nagdala ako ng payong.
Natutulog na ang sanggol sa kuna.
Gumagawa siya ng takdang-aralin bago matulog.
Naglinis kami ng bahay bago dumating ang bisita.
Kumain kami ng masarap na hapunan kagabi.
Nagtatrabaho ang tatay sa opisina araw-araw.
Kinain niya ang pagkain sa hapagkainan.
Pagkain ang pinakamasarap na bagay.
"""


def _write_corpus(tmpdir: str) -> str:
    """Write SMALL_CORPUS to a temp file and return its path."""
    path = os.path.join(tmpdir, "corpus.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SMALL_CORPUS)
    return path


# --------------------------------------------------------------------- #
#  Test class                                                             #
# --------------------------------------------------------------------- #


class TestBPEBasic(unittest.TestCase):
    """Low-level MorphAwareBPE tests."""

    def test_boundary_never_merged(self):
        """Merges must never combine tokens across a boundary marker."""
        corpus = [
            f"pag{BOUNDARY}kain",
            f"pag{BOUNDARY}kain",
            f"pag{BOUNDARY}kain",
            f"ma{BOUNDARY}ganda",
            f"ma{BOUNDARY}ganda",
        ]
        bpe = MorphAwareBPE()
        bpe.train(corpus, vocab_size=100)

        # No merge rule should produce a token containing the boundary
        for tok in bpe.vocab:
            if tok in MorphAwareBPE.SPECIALS or tok == BOUNDARY:
                continue
            self.assertNotIn(
                BOUNDARY, tok,
                f"Merge produced cross-boundary token: {tok!r}"
            )

    def test_save_and_load_roundtrip(self):
        """Save then load should produce identical encode results."""
        corpus = [f"um{BOUNDARY}kain", f"pag{BOUNDARY}kain"] * 5
        bpe = MorphAwareBPE()
        bpe.train(corpus, vocab_size=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            bpe.save(tmpdir)

            bpe2 = MorphAwareBPE()
            bpe2.load(tmpdir)

            self.assertEqual(bpe.vocab, bpe2.vocab)
            self.assertEqual(bpe.merges, bpe2.merges)

            text = f"um{BOUNDARY}kain"
            self.assertEqual(bpe.encode(text), bpe2.encode(text))


class TestTagalogTokenizerPipeline(unittest.TestCase):
    """End-to-end tokenizer tests."""

    @classmethod
    def setUpClass(cls):
        """Train a tokenizer once for all tests in this class."""
        cls._tmpdir = tempfile.mkdtemp()
        corpus_path = _write_corpus(cls._tmpdir)

        cls.tok = TagalogTokenizer()
        cls.tok.train(corpus_path, vocab_size=500)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    #  1. Round-trip: encode → decode ≈ original                          #
    # ------------------------------------------------------------------ #

    def test_roundtrip_simple_sentence(self):
        """encode then decode should return the original (lowercased) text."""
        text = "Kumain siya ng pagkain."
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        # Tokenizer lowercases; punctuation and spacing are preserved
        self.assertEqual(decoded, text.lower())

    def test_roundtrip_longer_sentence(self):
        text = "Ang mga bata ay masayang naglalaro sa labas."
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text.lower())

    # ------------------------------------------------------------------ #
    #  2. Morpheme consistency: "kain" root shared across inflections     #
    # ------------------------------------------------------------------ #

    def test_kain_root_consistency(self):
        """
        The root "kain" should be segmented out consistently across its
        inflections (kumain, pagkain, kinain).

        The segmenter decomposes:
            kumain  → [um, kain]  → surface: k▁um▁ain
            pagkain → [pag, kain] → surface: pag▁kain
            kinain  → [in, kain]  → surface: k▁in▁ain

        The key constraint: when "kain" appears as a direct morpheme
        (pagkain), BPE should encode it consistently.  For infixed forms,
        the root is split across the infix but the affix portions (um, in)
        and root fragments are still kept on their own side of boundaries.
        """
        # pagkain and bare "kain" should share the same root encoding
        # since the segmenter gives [pag, kain] → pag▁kain
        pagkain_ids = self.tok.encode("pagkain")
        pagkain_tokens = self.tok.tokenize("pagkain")

        # Encode bare "kain" (no affix, so no boundary)
        kain_ids = self.tok.bpe.encode("kain")

        # The last portion of pagkain's encoding should match bare "kain"
        # because the segmenter produces pag▁kain and BPE encodes each
        # side of the boundary independently
        self.assertEqual(
            pagkain_ids[-len(kain_ids):], kain_ids,
            f"'kain' IDs in pagkain {pagkain_ids} don't end with "
            f"bare kain IDs {kain_ids}"
        )

        # Also verify the segmenter finds "kain" in all three words
        for word in ["kumain", "pagkain", "kinain"]:
            morphemes = self.tok.segmenter.segment(word)
            self.assertIn("kain", morphemes,
                          f"Expected 'kain' in morphemes of {word}, got {morphemes}")

    # ------------------------------------------------------------------ #
    #  3. Fewer tokens than character-level baseline                      #
    # ------------------------------------------------------------------ #

    def test_fewer_tokens_than_char_level(self):
        """
        BPE tokenization should produce fewer tokens than a naive
        character-level tokenization for Filipino sentences.
        """
        sentences = [
            "Kumain siya ng pagkain sa hapagkainan.",
            "Maganda ang panahon ngayon kaya lumabas kami.",
            "Bumili ako ng mga prutas sa palengke kahapon.",
        ]

        for sentence in sentences:
            bpe_ids = self.tok.encode(sentence)
            # Character-level baseline: every non-space character is a token
            char_count = len(sentence.replace(" ", ""))
            self.assertLess(
                len(bpe_ids), char_count,
                f"BPE ({len(bpe_ids)} tokens) should be fewer than "
                f"char-level ({char_count} chars) for: {sentence!r}"
            )

    # ------------------------------------------------------------------ #
    #  4. Train, save, load, verify                                       #
    # ------------------------------------------------------------------ #

    def test_save_load_consistency(self):
        """Saved and re-loaded tokenizer should produce identical output."""
        with tempfile.TemporaryDirectory() as save_dir:
            self.tok.save(save_dir)

            tok2 = TagalogTokenizer()
            tok2.load(save_dir)

            text = "Nagluluto ang nanay ng masarap na adobo."
            self.assertEqual(self.tok.encode(text), tok2.encode(text))

    def test_tokenize_returns_strings(self):
        """tokenize() should return a list of subword strings."""
        tokens = self.tok.tokenize("kumain")
        self.assertIsInstance(tokens, list)
        for t in tokens:
            self.assertIsInstance(t, str)
        # Joining all tokens should reconstruct the word
        reconstructed = "".join(t for t in tokens if t != BOUNDARY)
        self.assertEqual(reconstructed, "kumain")

    def test_empty_input(self):
        """Empty input should return empty results."""
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")
        self.assertEqual(self.tok.tokenize(""), [])


class TestTagalogTokenizerEdgeCases(unittest.TestCase):
    """Edge-case tests (no training needed, or minimal training)."""

    def test_punctuation_preserved(self):
        """Punctuation should survive encode → decode."""
        tok = TagalogTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = _write_corpus(tmpdir)
            tok.train(corpus_path, vocab_size=300)

        text = "Kumain, nagluto, at naglinis."
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # All punctuation should be present
        for punct in [",", "."]:
            self.assertIn(punct, decoded,
                          f"Punctuation {punct!r} missing from decoded: {decoded!r}")

    def test_unknown_word_still_encodes(self):
        """Words not in training corpus should still encode (char fallback)."""
        tok = TagalogTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = _write_corpus(tmpdir)
            tok.train(corpus_path, vocab_size=300)

        ids = tok.encode("supercalifragilistic")
        self.assertTrue(len(ids) > 0, "Unknown word should still produce IDs")
        decoded = tok.decode(ids)
        # Should reconstruct something (might have <unk> for novel chars)
        self.assertTrue(len(decoded) > 0)


if __name__ == "__main__":
    unittest.main()
