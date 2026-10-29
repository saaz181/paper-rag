import unittest
import json
from pathlib import Path

from benchmarks.metrics import (
    METRIC_VERSION, NORMALIZATION, evidence_matches, exact_evidence_matches,
    fuzzy_contains, normalize_text,
)
from compare_runs import compatibility_issues


class NumericMatchingTests(unittest.TestCase):
    def assert_match(self, anchor, text, expected):
        case = {"expected_document": "paper.pdf", "expected_page": 0,
                "evidence": {"anchor": anchor}}
        result = {"document": "paper.pdf", "page": 0, "text": text}
        for matcher in (exact_evidence_matches, evidence_matches):
            with self.subTest(anchor=anchor, text=text, matcher=matcher.__name__):
                self.assertEqual(matcher(result, case), expected)

    def test_decimals_require_complete_numbers(self):
        for anchor, text, expected in (
            ("61", "The value was 61.9.", False),
            ("61", "The runtime was 61.9ms.", False),
            ("61", "The value was 161.", False),
            ("61.9", "The value was 61.99.", False),
            ("61.9", "The value was 61.9.", True),
            ("61", "The value was 61.", True),
            (".5", "The value was .59.", False),
            ("1,000", "There were 1000 examples.", True),
        ):
            self.assert_match(anchor, text, expected)

    def test_percentages_preserve_marker_and_decimal(self):
        for anchor, text, expected in (
            ("61%", "Accuracy was 61.9%.", False),
            ("61%", "Accuracy was 161%.", False),
            ("61%", "Accuracy was 61 %.", True),
            ("61.9%", "Accuracy was 61.9%.", True),
            ("61", "Accuracy was 61%.", False),
            ("61%", "There were 61 examples.", False),
        ):
            self.assert_match(anchor, text, expected)

    def test_negative_numbers_preserve_sign(self):
        for anchor, text, expected in (
            ("-61", "The change was 61.", False),
            ("61", "The change was -61.", False),
            ("-61", "The change was -61.9.", False),
            ("-61.9", "The change was \u221261.9.", True),
            ("-0.5%", "The change was -0.5 %.", True),
        ):
            self.assert_match(anchor, text, expected)

    def test_fuzzy_windows_preserve_complete_numbers(self):
        self.assertTrue(fuzzy_contains("61.9 percent accuracy", "61.9 percent accurcy"))
        self.assertFalse(fuzzy_contains("61.9 percent accuracy", "61.8 percent accurcy"))
        self.assertFalse(fuzzy_contains("-61.9 percent accuracy", "61.9 percent accurcy"))
        self.assertFalse(fuzzy_contains("61% accuracy", "61.9% accurcy"))
        self.assertFalse(fuzzy_contains("61 percent accuracy", "61.9 percent accuracy"))

    def test_pdf_cleanup_preserves_numbers(self):
        self.assertEqual(normalize_text("SEMAN-\ntically, \u221261.9 %; 1,000 examples."),
                         "semantically -61.9% 1000 examples")
        self.assertNotEqual(normalize_text("20- 30"), "2030")

    def test_scientific_notation_keeps_exponent_and_sign(self):
        self.assert_match("1e-3", "The rate was 1e-3.", True)
        self.assert_match("1e-3", "The rate was 1e-4.", False)
        self.assert_match("1", "The rate was 1e-3.", False)

    def test_comparator_rejects_previous_numeric_rules(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/baselines/minilm_150_25.json"
        previous = json.loads(path.read_text())
        current = json.loads(path.read_text())
        current["config"].update(metric_version=METRIC_VERSION, normalization=NORMALIZATION)
        issues = compatibility_issues(previous, current)
        self.assertTrue(any("metric_version" in issue for issue in issues))
        self.assertTrue(any("normalization" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
