"""Tagalog morphological tokenization module."""

from src.tagalog.tokenizer import TagalogTokenizer
from src.tagalog.segmenter import TagalogSegmenter
from src.tagalog.affixes import TagalogAffixes
from src.tagalog.roots import TagalogRoots
from src.tagalog.phonology import TagalogPhonology

__all__ = [
    "TagalogTokenizer",
    "TagalogSegmenter",
    "TagalogAffixes",
    "TagalogRoots",
    "TagalogPhonology",
]
