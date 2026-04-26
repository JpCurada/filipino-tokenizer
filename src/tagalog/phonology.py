class TagalogPhonology:
    """
    Phonological rules for Tagalog morphology.

    Nasal assimilation (pang-/mang- + root):
      + b/p  →  pam-/mam-  (root's first consonant dropped)
      + d/t/s  →  pan-/man-  (root's first consonant dropped)
      + k/g  →  pang-/mang-  (root's first consonant dropped)
      + vowel/h/l/m/n/w/y  →  pang-/mang-  (root unchanged)

    Suffix phonology:
      root ends in vowel + -an  →  surface -han
      root ends in vowel + -in  →  surface -hin
    """

    VOWELS = frozenset('aeiou')

    # Maps each surface nasal prefix to its canonical form and the consonants
    # that may have been dropped from the root's start ('' means nothing dropped).
    _NASAL_SURFACE = {
        'pam':  ('pang', ('b', 'p')),
        'mam':  ('mang', ('b', 'p')),
        'pan':  ('pang', ('d', 't', 's')),
        'man':  ('mang', ('d', 't', 's')),
        'pang': ('pang', ('k', 'g', '')),
        'mang': ('mang', ('k', 'g', '')),
    }

    # Consonants whose presence at root[0] triggers a specific nasal surface form.
    # Used by apply_nasal_assimilation.
    _NASAL_TRIGGER = {
        'b': 'm', 'p': 'm',          # pang → pam, mang → mam
        'd': 'n', 't': 'n', 's': 'n',  # pang → pan, mang → man
        'k': 'ng', 'g': 'ng',        # pang stays pang, mang stays mang (drop first)
    }

    # ------------------------------------------------------------------ #
    #  Nasal assimilation                                                  #
    # ------------------------------------------------------------------ #

    def apply_nasal_assimilation(self, prefix: str, root: str) -> tuple:
        """
        Given canonical prefix ('pang' or 'mang') and a root word, return the
        (surface_prefix, surface_root) pair after nasal assimilation.

        Examples:
            apply_nasal_assimilation('pang', 'bili')  → ('pam', 'ili')
            apply_nasal_assimilation('pang', 'sulat') → ('pan', 'ulat')
            apply_nasal_assimilation('pang', 'kamay') → ('pang', 'amay')
            apply_nasal_assimilation('pang', 'abot')  → ('pang', 'abot')
        """
        if not root:
            return (prefix, root)

        first = root[0].lower()
        is_pang = prefix.lower().startswith('p')
        base = 'pang' if is_pang else 'mang'

        trigger = self._NASAL_TRIGGER.get(first)

        if trigger == 'm':
            # b/p: prefix ending changes to -m, first consonant of root dropped
            surface_prefix = base[:-2] + 'm'   # 'pa' + 'm' = 'pam'
            return (surface_prefix, root[1:])
        elif trigger == 'n':
            # d/t/s: prefix ending changes to -n, first consonant of root dropped
            surface_prefix = base[:-1]          # 'pan' or 'man'
            return (surface_prefix, root[1:])
        elif trigger == 'ng':
            # k/g: prefix stays -ng, first consonant of root dropped
            return (base, root[1:])
        else:
            # vowel, h, l, m, n, w, y: no change to root
            return (base, root)

    def reverse_nasal_assimilation(self, surface_prefix: str, rest: str) -> list:
        """
        Given a surface nasal prefix and the portion of the word after it,
        return a list of (canonical_prefix, candidate_root) pairs representing
        all plausible original forms before assimilation.

        Examples:
            reverse_nasal_assimilation('pam', 'ili')
                → [('pang', 'bili'), ('pang', 'pili')]
            reverse_nasal_assimilation('pan', 'ulat')
                → [('pang', 'dulat'), ('pang', 'tulat'), ('pang', 'sulat')]
            reverse_nasal_assimilation('pang', 'amay')
                → [('pang', 'kamay'), ('pang', 'gamay'), ('pang', 'amay')]
        """
        info = self._NASAL_SURFACE.get(surface_prefix.lower())
        if info is None:
            return [(surface_prefix, rest)]

        canonical, dropped = info
        candidates = []
        for consonant in dropped:
            if consonant == '':
                candidates.append((canonical, rest))
            else:
                candidates.append((canonical, consonant + rest))
        return candidates

    def is_nasal_prefix(self, prefix: str) -> bool:
        """Return True if prefix is a surface nasal form of pang-/mang-."""
        return prefix.lower() in self._NASAL_SURFACE

    # ------------------------------------------------------------------ #
    #  Suffix phonology                                                    #
    # ------------------------------------------------------------------ #

    def needs_h_insertion(self, root: str) -> bool:
        """Return True if root ends in a vowel, requiring -h- before -an/-in."""
        return bool(root) and root[-1].lower() in self.VOWELS

    def apply_suffix_phonology(self, root: str, suffix: str) -> str:
        """
        Return the surface suffix to attach after root, inserting -h- when needed.

        Examples:
            apply_suffix_phonology('saya', 'an')  → 'han'
            apply_suffix_phonology('takot', 'an') → 'an'
            apply_suffix_phonology('huli', 'in')  → 'hin'
        """
        bare = suffix.lstrip('-')
        if bare == 'an' and self.needs_h_insertion(root):
            return 'han'
        if bare == 'in' and self.needs_h_insertion(root):
            return 'hin'
        return bare

    def strip_suffix(self, word: str, suffix: str) -> list:
        """
        Try to strip suffix from word, accounting for h-insertion variants.
        Returns a list of candidate root strings (may be empty if no match).

        Examples:
            strip_suffix('sayahan', 'an')  → ['saya']   (via -han)
            strip_suffix('takutan', 'an')  → ['takut']  (direct -an)
            strip_suffix('hulihin', 'in')  → ['huli']   (via -hin)
        """
        bare = suffix.lstrip('-')
        candidates = []

        if word.endswith(bare):
            candidates.append(word[:-len(bare)])

        if bare == 'an' and word.endswith('han'):
            root_candidate = word[:-3]      # strip 'han'; root ended in vowel
            if root_candidate not in candidates:
                candidates.append(root_candidate)
        elif bare == 'in' and word.endswith('hin'):
            root_candidate = word[:-3]      # strip 'hin'; root ended in vowel
            if root_candidate not in candidates:
                candidates.append(root_candidate)

        return candidates
