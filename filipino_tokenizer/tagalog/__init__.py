"""Tagalog morphological tokenization module."""

from filipino_tokenizer.tagalog.tokenizer import TagalogTokenizer
from filipino_tokenizer.tagalog.segmenter import TagalogSegmenter
from filipino_tokenizer.tagalog.affixes import TagalogAffixes
from filipino_tokenizer.tagalog.roots import TagalogRoots
from filipino_tokenizer.tagalog.phonology import TagalogPhonology

__all__ = [
    "TagalogTokenizer",
    "TagalogSegmenter",
    "TagalogAffixes",
    "TagalogRoots",
    "TagalogPhonology",
]
