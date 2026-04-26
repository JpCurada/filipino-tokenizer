import unittest
from src.tagalog.segmenter import TagalogSegmenter


class TestTagalogSegmenterSimple(unittest.TestCase):
    """Simple prefix/infix cases — single affix + root."""

    def setUp(self):
        self.seg = TagalogSegmenter()

    def test_kumain_um_infix(self):
        self.assertEqual(self.seg.segment('kumain'), ['um', 'kain'])

    def test_pagkain_pag_prefix(self):
        self.assertEqual(self.seg.segment('pagkain'), ['pag', 'kain'])

    def test_kinain_in_infix(self):
        self.assertEqual(self.seg.segment('kinain'), ['in', 'kain'])

    def test_maganda_ma_prefix(self):
        self.assertEqual(self.seg.segment('maganda'), ['ma', 'ganda'])


class TestTagalogSegmenterCircumfix(unittest.TestCase):
    """Circumfix cases — prefix + root + suffix detected as a unit."""

    def setUp(self):
        self.seg = TagalogSegmenter()

    def test_pagkainan_pag_an_circumfix(self):
        self.assertEqual(self.seg.segment('pagkainan'), ['pag', 'kain', 'an'])

    def test_kasiyahan_ka_han_circumfix(self):
        self.assertEqual(self.seg.segment('kasiyahan'), ['ka', 'siya', 'han'])


class TestTagalogSegmenterComplex(unittest.TestCase):
    """Complex stacked-prefix cases."""

    def setUp(self):
        self.seg = TagalogSegmenter()

    def test_pinakamahusay_stacked_prefixes(self):
        self.assertEqual(
            self.seg.segment('pinakamahusay'), ['pinaka', 'ma', 'husay']
        )


class TestTagalogSegmenterEdgeCases(unittest.TestCase):
    """Edge cases: frozen forms, loan words, function words, bare roots."""

    def setUp(self):
        self.seg = TagalogSegmenter()

    def test_pangalan_frozen_form(self):
        # 'pangalan' must NOT be segmented as pang- + alan
        # because both share identical definitions (frozen / duplicate entry)
        self.assertEqual(self.seg.segment('pangalan'), ['pangalan'])

    def test_computer_loan_word(self):
        self.assertEqual(self.seg.segment('computer'), ['computer'])

    def test_ang_function_word(self):
        self.assertEqual(self.seg.segment('ang'), ['ang'])

    def test_kain_bare_root(self):
        self.assertEqual(self.seg.segment('kain'), ['kain'])

    def test_empty_string(self):
        self.assertEqual(self.seg.segment(''), [])

    def test_case_insensitive(self):
        self.assertEqual(self.seg.segment('KUMAIN'), self.seg.segment('kumain'))


if __name__ == '__main__':
    unittest.main()
