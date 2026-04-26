"""
Training a Tagalog Tokenizer
=============================

Complete workflow: create a small corpus, train a morphology-aware BPE
tokenizer, save it to disk, and compare its output against a naive
character-level baseline.

Run from the project root:
    python examples/training_tagalog_tokenizer.py
"""

import os
import tempfile

from filipino_tokenizer.tagalog import TagalogTokenizer, TagalogSegmenter


# -------------------------------------------------------------------- #
#  1.  Prepare a small Filipino corpus                                  #
# -------------------------------------------------------------------- #

CORPUS = """\
Kumain siya ng pagkain sa hapagkainan.
Ang mga bata ay masayang naglalaro sa labas ng bahay.
Maganda ang panahon ngayon kaya lumabas kami para maglakad.
Bumili ako ng mga prutas sa palengke kahapon.
Nagluluto ang nanay ng masarap na adobo para sa buong pamilya.
Pumunta kami sa simbahan tuwing Linggo ng umaga.
Nagbabasa ang mga estudyante ng libro sa silid-aklatan.
Kumanta ang mga bata sa programa ng paaralan nila.
Umuulan kaya nagdala ako ng payong at kapote.
Naglinis kami ng bahay bago dumating ang mga bisita.
Nagtatrabaho ang tatay sa opisina araw-araw.
Kinain niya ang lahat ng pagkain sa mesa.
Magluluto ako ng sinigang na baboy mamaya.
Natutulog na ang sanggol sa kuna niya.
Gumagawa siya ng takdang-aralin bago matulog.
"""


def main():
    # Write corpus to a temp file
    tmpdir = tempfile.mkdtemp()
    corpus_path = os.path.join(tmpdir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(CORPUS)

    # ---------------------------------------------------------------- #
    #  2.  Train the tokenizer                                          #
    # ---------------------------------------------------------------- #

    print("=" * 64)
    print("  Filipino Tokenizer - Training Example")
    print("=" * 64)
    print()

    tok = TagalogTokenizer()
    print(f"Training on {corpus_path} ...")
    tok.train(corpus_path, vocab_size=500)
    print(f"Vocabulary size: {len(tok.bpe.vocab)}")
    print(f"Learned merges:  {len(tok.bpe.merges)}")
    print()

    # ---------------------------------------------------------------- #
    #  3.  Save and reload                                              #
    # ---------------------------------------------------------------- #

    save_dir = os.path.join(tmpdir, "tokenizer")
    tok.save(save_dir)
    print(f"Saved tokenizer to {save_dir}/")

    tok2 = TagalogTokenizer()
    tok2.load(save_dir)
    print("Reloaded tokenizer from disk - OK")
    print()

    # ---------------------------------------------------------------- #
    #  4.  Encode / decode examples                                     #
    # ---------------------------------------------------------------- #

    examples = [
        "Kumain siya ng pagkain.",
        "Maganda ang panahon ngayon.",
        "Nagluluto ang nanay ng masarap na adobo.",
        "Bumili ako ng mga prutas sa palengke.",
    ]

    print("-" * 64)
    print("  Encode / Decode Round-trip")
    print("-" * 64)
    for text in examples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        tokens = tok.tokenize(text)
        print(f"  Input:   {text}")
        print(f"  Tokens:  {tokens}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded}")
        print()

    # ---------------------------------------------------------------- #
    #  5.  Morpheme-aware vs naive character-level comparison            #
    # ---------------------------------------------------------------- #

    print("-" * 64)
    print("  Morpheme-Aware BPE  vs  Character-Level Baseline")
    print("-" * 64)
    print()

    seg = TagalogSegmenter()

    for text in examples:
        bpe_tokens = tok.tokenize(text)
        char_tokens = list(text.replace(" ", ""))

        bpe_count = len(bpe_tokens)
        char_count = len(char_tokens)
        savings = ((char_count - bpe_count) / char_count) * 100

        print(f"  Sentence: {text}")
        print(f"    Morphemes:    {seg.segment_text(text)}")
        print(f"    BPE tokens:   {bpe_count:3d}  {bpe_tokens}")
        print(f"    Char tokens:  {char_count:3d}  (one per character)")
        print(f"    Savings:      {savings:.0f}% fewer tokens with BPE")
        print()

    # ---------------------------------------------------------------- #
    #  6.  Show how the root "kain" is shared across inflections        #
    # ---------------------------------------------------------------- #

    print("-" * 64)
    print("  Root Sharing: 'kain' across inflected forms")
    print("-" * 64)
    print()

    kain_words = ["kain", "kumain", "pagkain", "kinain"]
    for word in kain_words:
        morphemes = seg.segment(word)
        tokens = tok.tokenize(word)
        ids = tok.encode(word)
        print(f"  {word:12s}  morphemes={morphemes!s:24s}  "
              f"tokens={tokens}  ids={ids}")

    print()
    print("=" * 64)
    print("  Done.")
    print("=" * 64)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
