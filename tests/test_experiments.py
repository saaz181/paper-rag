import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_runs import compatibility_issues, load_report
from evaluate import analyze_case, summarize
from src.lexical import BM25Index
from src.retriever import retrieve_hybrid, reciprocal_rank_fusion
from test_compare_runs import make_report
from run_reranker_experiments import candidate_changes, project_top_three


class ExperimentTests(unittest.TestCase):
    def test_rank_five_is_hit_five_but_not_hit_three(self):
        case = {"id": "example", "question": "q", "category": "method",
                "expected_document": "p", "expected_page": 1,
                "evidence": {"anchor": "useful evidence"}}
        wrong = {"document": "p", "page": 1, "text": "other passage"}
        good = {**wrong, "text": "useful evidence"}
        results = [dict(wrong) for _ in range(4)] + [good]
        row = analyze_case(case, results, results, results, final_top_k=5)
        scores = summarize([row])["evidence"]
        self.assertEqual(scores["hit_at_3"], 0)
        self.assertEqual(scores["hit_at_5"], 1)
        self.assertEqual(scores["mrr_at_3"], 0)
        self.assertEqual(scores["mrr_at_5"], .2)
        self.assertIsNone(row["failure_type"])
        short = analyze_case(case, results, results, results[:3])
        self.assertNotIn("evidence_hit_5", short)
        self.assertEqual(short["failure_type"], "reranking_failure")

    def test_top_five_report_validation_and_declared_reranker_variable(self):
        report = make_report()
        report["config"].update(rerank_top_k=5, mrr_cutoff=5, failure_cutoff=5)
        for group in [report["overall"], *report["categories"].values()]:
            for metric in ("page", "strict_evidence", "exact_evidence", "evidence"):
                group[metric]["hit_at_5"] = group[metric]["hit_at_3"]
        for case in report["cases"]:
            case["evidence_hit_5"] = case["evidence_hit_3"]
        report["overall"]["evidence"]["hit_at_5"] = 1
        report["overall"]["failure_counts"]["evidence"] = {
            key: 0 for key in report["overall"]["failure_counts"]["evidence"]}
        category = report["cases"][1]["category"]
        report["categories"][category]["evidence"]["hit_at_5"] = 1
        report["categories"][category]["failure_counts"]["evidence"] = {
            key: 0 for key in report["categories"][category]["failure_counts"]["evidence"]}
        report["cases"][1]["evidence_hit_5"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(load_report(path), report)
        other = {**report, "config": {**report["config"], "reranker_model": "other"}}
        self.assertTrue(compatibility_issues(report, other))
        self.assertEqual(compatibility_issues(report, other, vary=["reranker_model"]), [])

    def test_rrf_constant_can_change_membership(self):
        def chunk(i):
            return {"document": "p", "page": 1, "chunk_id": i, "text": "text"}
        dense = [chunk(i) for i in range(20)]
        lexical = [chunk(i) for i in range(30, 48)] + [chunk(19)]
        small = reciprocal_rank_fusion(dense, lexical, top_k=1, rrf_k=1)
        large = reciprocal_rank_fusion(dense, lexical, top_k=1, rrf_k=100)
        self.assertEqual(small[0]["chunk_id"], 0)
        self.assertEqual(large[0]["chunk_id"], 19)

    def test_diagnostics_counts_and_averages(self):
        import numpy as np
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.]])
        chunks = [{"document": "p", "page": 1, "chunk_id": i, "text": text}
                  for i, text in enumerate(("background", "colbert"))]
        results, diagnostics = retrieve_hybrid("colbert", Encoder(), chunks, np.eye(2),
                                              BM25Index(chunks), branch_top_k=1,
                                              return_diagnostics=True)
        self.assertEqual(diagnostics["candidate_counts"],
                         {"dense": 1, "bm25": 1, "overlap": 0, "union": 2, "reranker_input": 2})
        self.assertEqual(len(results), 2)
        self.assertTrue(all(value >= 0 for value in diagnostics["latency_ms"].values()))

    def test_candidate_changes_distinguishes_order_and_membership(self):
        a = {"document": "p", "page": 1, "chunk_id": 1}
        b = {**a, "chunk_id": 2}
        c = {**a, "chunk_id": 3}
        base = {"cases": [{"id": "q", "candidates": [a, b]}]}
        reordered = {"cases": [{"id": "q", "candidates": [b, a]}]}
        changed = {"cases": [{"id": "q", "candidates": [a, c]}]}
        order = candidate_changes(base, reordered)
        self.assertEqual(order["membership_changed"], 0)
        self.assertEqual(order["order_changed"], 1)
        self.assertEqual(order["average_jaccard"], 1)
        membership = candidate_changes(base, changed)
        self.assertEqual(membership["membership_changed"], 1)
        self.assertAlmostEqual(membership["average_jaccard"], 1/3)

    def test_top_three_projection_reclassifies_failures(self):
        case = {"id": "q", "question": "q", "category": "method",
                "expected_document": "p", "expected_page": 0,
                "evidence": {"anchor": "useful evidence"}}
        chunks = [{"document": "p", "page": 0, "chunk_id": i,
                   "text": "useful evidence" if i == 4 else "unrelated text"}
                  for i in range(5)]
        row = analyze_case(case, chunks, chunks, chunks, final_top_k=5)
        row.update(candidates=chunks, candidate_counts={"reranker_input": 5}, latency_ms={"total": 12})
        report = {"config": {"page_base": 0, "fuzzy_threshold": .8}, "cases": [row]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "chunks.json").write_text(json.dumps(chunks))
            with patch("run_reranker_experiments.load_benchmark", return_value=[case]):
                projected = project_top_three(report, path)
        self.assertEqual(projected["overall"]["evidence"]["hit_at_3"], 0)
        self.assertNotIn("hit_at_5", projected["overall"]["evidence"])
        self.assertEqual(projected["cases"][0]["failure_type"], "reranking_failure")
        self.assertEqual(projected["overall"]["average_latency_ms"]["total"], 12)


if __name__ == "__main__":
    unittest.main()
