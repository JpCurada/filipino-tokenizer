"""
HuggingFace PreTrainedTokenizer wrapper for TagalogTokenizer.

Allows the tokenizer to plug directly into HuggingFace Trainer, TRL,
Axolotl, and any other transformers-based training pipeline.

Requires: pip install transformers
  or:     pip install filipino-tokenizer[hf]

Usage
-----
Load from a local directory::

    from filipino_tokenizer.tagalog import TagalogHFTokenizer
    tok = TagalogHFTokenizer.from_pretrained("demo/models/morph/")

Load from HuggingFace Hub (after uploading)::

    tok = TagalogHFTokenizer.from_pretrained("<your-hf-username>/tagalog-tokenizer")

Use in HF Trainer::

    from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok)))
    trainer = Trainer(model=model, tokenizer=tok, ...)
"""

import os

try:
    from transformers import PreTrainedTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    # Catches ImportError, ModuleNotFoundError, and any error from a partial
    # transformers install (AttributeError, etc.), so the class definition
    # always succeeds without silently crippling the MRO.
    _TRANSFORMERS_AVAILABLE = False

    class PreTrainedTokenizer:  # type: ignore[no-redef]
        """Placeholder so the class body can be parsed without transformers."""
        def __init__(self, *args, **kwargs):
            pass

from filipino_tokenizer.tagalog.tokenizer import TagalogTokenizer


class TagalogHFTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer for Tagalog.

    Wraps ``TagalogTokenizer`` (morphological segmentation + constrained BPE)
    behind the ``PreTrainedTokenizer`` interface so it works with the full
    HuggingFace ecosystem.

    Special tokens (already part of the BPE vocab):
        ``<pad>``  id=0
        ``<unk>``  id=1
        ``<s>``    id=2
        ``</s>``   id=3
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    # GPT-style models don't need token_type_ids; keeping model inputs minimal
    # avoids accidental None token-type fields during padding.
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        merges_file: str | None = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for TagalogHFTokenizer.\n"
                "Install it with:  pip install transformers\n"
                "or:               pip install filipino-tokenizer[hf]"
            )

        self._inner = TagalogTokenizer()
        if vocab_file is None:
            # Load the pretrained model bundled with the package
            self._inner.load_pretrained()
        else:
            # MorphAwareBPE.load() expects a directory; derive it from vocab_file
            model_dir = os.path.dirname(os.path.abspath(vocab_file))
            self._inner.bpe.load(model_dir)

        vocab = self._inner.bpe.vocab
        fallback_unk = "<unk>" if "<unk>" in vocab else next(iter(vocab))
        fallback_pad = "<pad>" if "<pad>" in vocab else fallback_unk

        # Ensure special tokens always resolve to real IDs. Older/custom saved
        # vocabs may not include all expected special strings.
        if unk_token not in vocab:
            unk_token = fallback_unk
        if pad_token not in vocab:
            pad_token = fallback_pad
        if bos_token not in vocab:
            bos_token = None
        if eos_token not in vocab:
            eos_token = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Required PreTrainedTokenizer interface                              #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._inner.bpe.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._inner.bpe.vocab)

    def _tokenize(self, text: str) -> list[str]:
        """Morphological segmentation + constrained BPE → token strings."""
        return self._inner.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        vocab = self._inner.bpe.vocab
        # self.unk_token is an AddedToken after super().__init__(); str() extracts
        # the content string so the plain-dict lookup works in all HF versions.
        unk_str = str(self.unk_token) if self.unk_token is not None else "<unk>"
        unk_id = vocab.get(unk_str, 1)
        if not isinstance(unk_id, int):
            unk_id = 1
        if token is None:
            return unk_id
        result = vocab.get(str(token), unk_id)
        # Guard: vocab values must be int; None would crash HF padding logic.
        return result if isinstance(result, int) else unk_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._inner.bpe.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Decode a list of token strings back to readable text."""
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self._inner.bpe.decode(ids)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, str]:
        """Save vocab.json and merges.txt to *save_directory*."""
        os.makedirs(save_directory, exist_ok=True)
        self._inner.save(save_directory)
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.txt")
        return (vocab_file, merges_file)
