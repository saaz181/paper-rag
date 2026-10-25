import unittest

from benchmarks.metrics import (
    evidence_matches, exact_evidence_matches, fuzzy_contains, hit_at_k,
    normalize_text, page_matches, reciprocal_rank, strict_evidence_matches,
)
from benchmarks.test_cases import load_benchmark, load_test_cases
from evaluate import analyze_case, classify_failure, detect_page_base, summarize, validate_cases


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "example", "question": "What is the result?",
            "expected_document": "paper.pdf", "expected_page": 0,
            "category": "method", "evidence": {"anchor": "efficient retrieval"},
        }
        self.good = {"document": "paper.pdf", "page": 0,
                     "text": "Uses EFFICIENT\n retrieval."}
        self.wrong_evidence = {**self.good, "text": "Unrelated passage."}
        self.wrong_page = {**self.good, "page": 1}
        self.documents = {"paper.pdf": {"pages": [{"page": 0}]}}

    def test_full_loader_preserves_evaluation_labels(self):
        cases = load_benchmark()
        self.assertEqual(len(cases), 301)
        for case in cases:
            self.assertTrue(case["id"])
            self.assertTrue(case["category"])
            self.assertTrue(case["evidence"]["anchor"])
        # Keep the intentionally minimal public loader backward compatible.
        self.assertEqual(set(load_test_cases()[0]),
                         {"question", "expected_document", "expected_page"})

    def test_page_base_matches_ingested_documents(self):
        for base in (0, 1):
            documents = {"paper.pdf": {"pages": [{"page": base}, {"page": base + 1}]}}
            self.assertEqual(detect_page_base(documents), base)
            self.assertEqual(load_benchmark(base)[0]["expected_page"], base)
        with self.assertRaises(ValueError):
            detect_page_base({})
        with self.assertRaises(ValueError):
            detect_page_base({**self.documents, "other.pdf": {"pages": [{"page": 1}]}})

    def test_incomplete_labels_are_rejected(self):
        validate_cases([self.case], self.documents)
        for change in ({"evidence": {}}, {"category": ""}, {"expected_page": 9}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cases([{**self.case, **change}], self.documents)
        with self.assertRaises(ValueError):
            validate_cases([], self.documents)

    def test_evidence_normalization_and_document_constraint(self):
        self.assertTrue(evidence_matches(self.good, self.case))
        self.assertFalse(evidence_matches(self.wrong_evidence, self.case))
        self.assertTrue(evidence_matches(self.wrong_page, self.case))
        self.assertTrue(exact_evidence_matches(self.wrong_page, self.case))
        self.assertFalse(strict_evidence_matches(self.wrong_page, self.case))
        self.assertFalse(evidence_matches({**self.good, "document": "other.pdf"}, self.case))
        case = {**self.case, "evidence": {"anchor": "efficient pre-computation"}}
        self.assertTrue(evidence_matches({**self.good, "text": "efﬁcient pre-\ncomputation"}, case))
        self.assertFalse(evidence_matches(self.good, {**case, "evidence": {"anchor": "---"}}))

    def test_metrics_distinguish_page_and_evidence_ranks(self):
        results = [self.wrong_evidence, self.wrong_page, self.good]
        self.assertEqual(hit_at_k(results, self.case, 1, page_matches), 1)
        self.assertEqual(hit_at_k(results, self.case, 1, evidence_matches), 0)
        self.assertEqual(hit_at_k(results, self.case, 3, evidence_matches), 1)
        self.assertAlmostEqual(reciprocal_rank(results, self.case, evidence_matches), 1 / 2)
        self.assertAlmostEqual(reciprocal_rank(results, self.case, strict_evidence_matches), 1 / 3)

    def test_pdf_normalization_and_empty_anchors(self):
        self.assertEqual(normalize_text("SEMAN- \ntically efﬁcient, retrieval!"),
                         "semantically efficient retrieval")
        self.assertFalse(fuzzy_contains("", "anything"))
        self.assertFalse(fuzzy_contains("---", "anything"))
        self.assertFalse(fuzzy_contains("retrieval evidence", ""))
        self.assertFalse(fuzzy_contains("retrieval evidence", "retrieval"))

    def test_fuzzy_threshold_and_numeric_guards(self):
        self.assertTrue(fuzzy_contains("efficient retrieval", "efficient retrievel"))
        self.assertFalse(fuzzy_contains("efficient retrieval", "efficient retrievel", 1.0))
        self.assertFalse(fuzzy_contains("49 995 000", "49 995 001"))
        self.assertFalse(fuzzy_contains("44.1", "44.9"))
        self.assertFalse(fuzzy_contains("61", "161"))
        self.assertFalse(fuzzy_contains("61 percent accuracy", "62 percent accuracy"))
        self.assertTrue(fuzzy_contains("61 percent accuracy", "61 percent accurcy"))
        self.assertFalse(exact_evidence_matches(
            {**self.good, "text": "inefficient retrieval"}, self.case))
        for threshold in (0, -0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                fuzzy_contains("a", "a", threshold)

    def test_document_wide_chunk_audit(self):
        # An anchor retained on a neighboring page is not a chunking failure.
        self.assertIsNone(classify_failure(self.case, [self.wrong_page],
                                          [self.wrong_page], [self.wrong_page]))
        self.assertEqual(classify_failure(self.case, [self.wrong_page],
                                          [self.wrong_page], [self.wrong_page],
                                          strict_evidence_matches), "chunking_failure")

    def test_case_and_summary_use_consistent_matchers(self):
        result = {**self.wrong_page, "text": "efficient retrievel"}
        row = analyze_case(self.case, [result], [result], [result])
        self.assertFalse(row["page_hit_1"])
        self.assertFalse(row["strict_evidence_hit_1"])
        self.assertFalse(row["exact_evidence_hit_1"])
        self.assertTrue(row["evidence_hit_1"])
        self.assertIsNone(row["failure_type"])
        self.assertEqual(row["failure_types"]["exact_evidence"], "chunking_failure")
        self.assertTrue(row["retrieved_results"][0]["matches"]["evidence"])
        summary = summarize([row])
        self.assertEqual(summary["evidence"]["mrr"], 1)
        self.assertEqual(sum(summary["failure_counts"]["evidence"].values()), 0)
        strict_row = analyze_case(self.case, [result], [result], [result], threshold=1)
        self.assertFalse(strict_row["evidence_hit_1"])
        with self.assertRaises(ValueError):
            summarize([])

    def test_first_stage_failure_and_success(self):
        scenarios = [
            ([], [], [], "chunking_failure"),
            ([self.good], [self.wrong_evidence], [self.wrong_evidence], "retrieval_failure"),
            ([self.good], [self.good], [self.wrong_evidence], "reranking_failure"),
            ([self.good], [self.good], [self.good], None),
        ]
        for chunks, candidates, results, expected in scenarios:
            with self.subTest(expected=expected):
                self.assertEqual(classify_failure(self.case, chunks, candidates, results), expected)

    def test_current_chunks_override_stale_audit(self):
        case = {**self.case, "chunk_audit": {"anchor_retained": False}}
        self.assertIsNone(classify_failure(case, [self.good], [self.good], [self.good]))
        case["chunk_audit"]["anchor_retained"] = True
        self.assertEqual(classify_failure(case, [], [], []), "chunking_failure")


if __name__ == "__main__":
    unittest.main()
