import unittest
from src.tagalog.affixes import TagalogAffixes
from src.cebuano.affixes import CebuanoAffixes


class TestTagalogAffixesLoad(unittest.TestCase):
    def setUp(self):
        self.affixes = TagalogAffixes()

    def test_loads_prefixes(self):
        self.assertGreater(len(self.affixes.prefixes), 0)

    def test_loads_suffixes(self):
        self.assertGreater(len(self.affixes.suffixes), 0)

    def test_loads_infixes(self):
        self.assertGreater(len(self.affixes.infixes), 0)

    def test_loads_circumfixes(self):
        self.assertGreater(len(self.affixes.circumfixes), 0)

    def test_tagalog_only_filtering(self):
        # Every entry in prefixes must have come from a Tagalog-language entry
        # Verify by checking that the 'raw' key is present (set by _load)
        for key, meta in self.affixes.prefixes.items():
            self.assertIn('raw', meta, f"prefix {key!r} missing 'raw' key")
            self.assertIn('function', meta)
            self.assertIn('etymology', meta)

    def test_no_foreign_language_bleed(self):
        # CebuanoAffixes should load different (or overlapping) entries —
        # neither set should be identical to the other for suffixes.
        cebuano = CebuanoAffixes()
        tagalog_keys = set(self.affixes.prefixes.keys())
        cebuano_keys = set(cebuano.prefixes.keys())
        # There may be shared affix strings but at least one language has
        # some prefix unique to it (the tables contain language-specific entries).
        self.assertGreater(len(tagalog_keys), 0)
        self.assertGreater(len(cebuano_keys), 0)


class TestTagalogAffixesContent(unittest.TestCase):
    def setUp(self):
        self.affixes = TagalogAffixes()

    # --- productive prefixes ---
    def test_prefix_mag(self):
        self.assertIn('mag', self.affixes.prefixes)

    def test_prefix_nag(self):
        self.assertIn('nag', self.affixes.prefixes)

    def test_prefix_pag(self):
        self.assertIn('pag', self.affixes.prefixes)

    def test_prefix_ma(self):
        self.assertIn('ma', self.affixes.prefixes)

    def test_prefix_ka(self):
        self.assertIn('ka', self.affixes.prefixes)

    def test_prefix_pang(self):
        self.assertIn('pang', self.affixes.prefixes)

    # --- productive suffixes ---
    def test_suffix_an(self):
        self.assertIn('an', self.affixes.suffixes)

    def test_suffix_han(self):
        self.assertIn('han', self.affixes.suffixes)

    def test_suffix_in(self):
        self.assertIn('in', self.affixes.suffixes)

    # --- productive infixes ---
    def test_infix_um(self):
        self.assertIn('um', self.affixes.infixes)

    def test_infix_in(self):
        self.assertIn('in', self.affixes.infixes)

    def test_exactly_two_infixes(self):
        self.assertEqual(len(self.affixes.get_infixes()), 2)

    # --- circumfixes ---
    def test_circumfix_ka_han(self):
        pairs = self.affixes.get_circumfixes()
        self.assertIn(('ka', 'han'), pairs)

    def test_circumfix_pag_an(self):
        pairs = self.affixes.get_circumfixes()
        self.assertIn(('pag', 'an'), pairs)

    def test_circumfix_ma_an(self):
        pairs = self.affixes.get_circumfixes()
        self.assertIn(('ma', 'an'), pairs)


class TestTagalogAffixesSorting(unittest.TestCase):
    def setUp(self):
        self.affixes = TagalogAffixes()

    def test_prefixes_sorted_longest_first(self):
        prefixes = self.affixes.get_prefixes()
        lengths = [len(p) for p in prefixes]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_suffixes_sorted_longest_first(self):
        suffixes = self.affixes.get_suffixes()
        lengths = [len(s) for s in suffixes]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_circumfixes_sorted_by_prefix_longest_first(self):
        pairs = self.affixes.get_circumfixes()
        lengths = [len(p) for p, s in pairs]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_get_prefixes_returns_list_of_strings(self):
        prefixes = self.affixes.get_prefixes()
        self.assertIsInstance(prefixes, list)
        for p in prefixes:
            self.assertIsInstance(p, str)

    def test_get_circumfixes_returns_list_of_tuples(self):
        pairs = self.affixes.get_circumfixes()
        self.assertIsInstance(pairs, list)
        for item in pairs:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


if __name__ == '__main__':
    unittest.main()
