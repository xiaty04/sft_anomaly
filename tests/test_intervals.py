import unittest

from tsad_v2.intervals import canonicalize, parse_interval_output, to_json


class IntervalTests(unittest.TestCase):
    def test_parse_strict_json_with_surrounding_text(self):
        result = parse_interval_output('answer: [{"start": 3, "end": 7}] done', 0, 10)
        self.assertTrue(result.valid)
        self.assertEqual(result.intervals, [(3, 7)])

    def test_empty_and_touching_intervals(self):
        self.assertEqual(parse_interval_output("[]", 0, 10).intervals, [])
        self.assertEqual(canonicalize([(1, 3), (3, 5)]), [(1, 5)])

    def test_rejects_extra_fields_and_bad_bounds(self):
        extra = parse_interval_output('[{"start":1,"end":2,"label":"x"}]', 0, 10)
        reverse = parse_interval_output('[{"start":5,"end":2}]', 0, 10)
        self.assertFalse(extra.valid)
        self.assertFalse(reverse.valid)

    def test_json_output_is_compact(self):
        self.assertEqual(to_json([(1, 2)]), '[{"start":1,"end":2}]')


if __name__ == "__main__":
    unittest.main()

