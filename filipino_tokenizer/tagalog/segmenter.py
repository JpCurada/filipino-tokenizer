from filipino_tokenizer.base import BaseSegmenter
from filipino_tokenizer.tagalog.affixes import TagalogAffixes
from filipino_tokenizer.tagalog.roots import TagalogRoots
from filipino_tokenizer.tagalog.phonology import TagalogPhonology


class TagalogSegmenter(BaseSegmenter):
    """
    Multi-pass morphological segmenter for Tagalog.

    Pass order (per SKILL.md):
      0. Frozen-form guard     — words whose affix analysis is blocked by
                                 identical-definition duplicates in the dict.
      1. Circumfix detection   — ka- -han, pag- -an, etc.
      2. Prefix stripping      — longest-match-first, recursive for stacked prefixes
      3. Infix detection       — -um- and -in- after first consonant
      4. Suffix stripping      — -an/-han, -in/-hin phonology variants
      5. Fallback              — return [word] unsegmented

    Root validation:  every candidate root is checked against the root
    dictionary before a segmentation is accepted.

    Redundancy check: if both the whole word and the stripped root appear in
    the dictionary with identical definitions the analysis is rejected.
    This catches frozen forms like 'pangalan' where 'alan' and 'pangalan'
    share the same definition ("name; reputation; repute; denomination").

    _MIN_ROOT = 4: roots shorter than 4 characters are rejected to avoid
    spurious matches against short dictionary fragments (e.g. 'gka', 'nda')
    that appear as roots only because the dictionary stores inflected forms
    under truncated keys.
    """

    VOWELS = frozenset('aeiou')
    _MIN_ROOT = 4   # minimum characters a valid root must have

    def __init__(self):
        self.affixes = TagalogAffixes()
        self.roots = TagalogRoots()
        self.phonology = TagalogPhonology()

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def segment(self, word: str) -> list:
        word = word.lower().strip()
        if not word:
            return []

        # Guard: frozen/lexicalized forms are returned unsegmented
        if self._is_frozen(word):
            return [word]

        return (
            self._try_circumfix(word) or
            self._try_prefix(word) or
            self._try_infix(word) or
            self._try_suffix(word) or
            [word]
        )

    # ------------------------------------------------------------------ #
    #  Pass 0 — frozen-form guard                                          #
    # ------------------------------------------------------------------ #

    def _is_frozen(self, word: str) -> bool:
        """
        Return True if the word is a frozen/lexicalized form that should not
        be morphologically decomposed.

        A word is frozen when it is itself in the roots dictionary AND at
        least one prefix-stripping yields a root whose dictionary definition
        is identical to the whole word's definition.  That identity signals
        a duplicate/alternate-form entry rather than productive affixation.

        Example: 'pangalan' (name) → strip 'pang-' → 'alan' (name).
        Both share the same definition, so the analysis is frozen.
        """
        if not self.roots.is_root(word):
            return False
        for prefix in self.affixes.get_prefixes():
            p = len(prefix)
            if len(word) <= p + self._MIN_ROOT - 1:
                continue
            if word[:p] != prefix:
                continue
            remainder = word[p:]
            if (self.roots.is_root(remainder)
                    and self._is_redundant(word, remainder)):
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Pass 1 — circumfix                                                  #
    # ------------------------------------------------------------------ #

    def _try_circumfix(self, word: str) -> list | None:
        for prefix, suffix in self.affixes.get_circumfixes():
            p, s = len(prefix), len(suffix)
            if len(word) <= p + s:
                continue
            if word[:p] != prefix or word[-s:] != suffix:
                continue
            core = word[p:-s]
            if (len(core) >= self._MIN_ROOT
                    and self.roots.is_root(core)
                    and not self._is_redundant(word, core)):
                return [prefix, core, suffix]
        return None

    # ------------------------------------------------------------------ #
    #  Pass 2 — prefix (recursive for stacked)                            #
    # ------------------------------------------------------------------ #

    def _try_prefix(self, word: str, depth: int = 0) -> list | None:
        if depth > 3:
            return None

        for prefix in self.affixes.get_prefixes():
            p = len(prefix)
            # Remainder must be at least _MIN_ROOT chars
            if len(word) <= p + self._MIN_ROOT - 1:
                continue
            if word[:p] != prefix:
                continue

            remainder = word[p:]

            # Try deeper segmentation of remainder before accepting bare root
            sub = self._try_prefix(remainder, depth + 1)
            if sub:
                return [prefix] + sub

            sub = self._try_infix(remainder)
            if sub:
                return [prefix] + sub

            # Accept remainder as a bare root
            if (self.roots.is_root(remainder)
                    and not self._is_redundant(word, remainder)):
                return [prefix, remainder]

        return None

    # ------------------------------------------------------------------ #
    #  Pass 3 — infix                                                      #
    # ------------------------------------------------------------------ #

    def _try_infix(self, word: str) -> list | None:
        # Infixes attach after the first consonant only
        if len(word) < 3 or word[0] in self.VOWELS:
            return None
        first = word[0]
        for infix in self.affixes.get_infixes():
            n = len(infix)
            if word[1:1 + n] == infix:
                root = first + word[1 + n:]
                if len(root) >= self._MIN_ROOT and self.roots.is_root(root):
                    return [infix, root]
        return None

    # ------------------------------------------------------------------ #
    #  Pass 4 — suffix                                                     #
    # ------------------------------------------------------------------ #

    def _try_suffix(self, word: str) -> list | None:
        for suffix in self.affixes.get_suffixes():
            for root_cand in self.phonology.strip_suffix(word, suffix):
                if (len(root_cand) >= self._MIN_ROOT
                        and self.roots.is_root(root_cand)
                        and not self._is_redundant(word, root_cand)):
                    surface_suf = self.phonology.apply_suffix_phonology(
                        root_cand, suffix
                    )
                    return [root_cand, surface_suf]
        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _is_redundant(self, word: str, root_candidate: str) -> bool:
        """
        Return True when the whole word and the candidate root appear in the
        roots dictionary with identical definitions.
        """
        info_w = self.roots.get_root_info(word)
        info_r = self.roots.get_root_info(root_candidate)
        if info_w and info_r:
            return info_w['definition'].strip() == info_r['definition'].strip()
        return False
