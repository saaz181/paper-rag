import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_runs import compare_reports, compatibility_issues, load_report
from evaluate import (
    EMBEDDING_MODEL_NAME, CHUNK_SIZE, OVERLAP, QueryEncoder, analyze_case,
    embedding_prefixes, embedding_signature, encode_passages, summarize,
)
from src.cache import create_signature

ROOT = Path(__file__).resolve().parents[1]


def make_report(hits=(True, False), model="all-MiniLM-L6-v2"):
    config = json.loads((ROOT / "benchmarks/baselines/minilm_150_25.json").read_text())["config"]
    config["embedding_model"] = model
    rows = []
    for index, hit in enumerate(hits):
        case = {"id": f"case_{index}", "question": f"Question {index}?",
                "category": ("concept", "dataset")[index],
                "expected_document": "paper.pdf", "expected_page": 0,
                "evidence": {"anchor": "efficient retrieval"}}
        good = {"document": "paper.pdf", "page": 0, "text": "efficient retrieval"}
        wrong = {**good, "text": "unrelated passage"}
        results = [good if hit else wrong]
        rows.append(analyze_case(case, [good], results, results))
    return {"config": config, "overall": summarize(rows),
            "categories": {row["category"]: summarize([row]) for row in rows}, "cases": rows}


class CompareRunsTests(unittest.TestCase):
    def test_changes_categories_and_paired_cases(self):
        baseline = make_report()
        candidate = make_report((False, True), "intfloat/e5-small-v2")
        self.assertEqual(compatibility_issues(baseline, candidate), [])
        output = compare_reports(baseline, candidate)
        self.assertIn("0.500 → 0.500  (+0.000)", output)
        self.assertIn("1 gained, 1 lost, 0 unchanged", output)
        self.assertIn("concept (N: 1 → 1)", output)
        self.assertIn("dataset (N: 1 → 1)", output)
        self.assertIn("0.000 → 1.000  (+1.000)", output)
        self.assertIn("1.000 → 0.000  (-1.000)", output)
        self.assertIn("retrieval failures:", output)

    def test_ignores_model_dependent_signature(self):
        baseline, candidate = make_report(), make_report()
        candidate["config"].update(corpus_signature="different", query_prefix="query: ",
                                    passage_prefix="passage: ", embedding_dimension=768)
        self.assertEqual(compatibility_issues(baseline, candidate), [])

    def test_rejects_fixed_setting_and_data_mismatches(self):
        baseline = make_report()
        for key, value in (("fuzzy_threshold", .85), ("retrieval_top_k", 50),
                           ("chunks_sha256", "changed"), ("benchmark_sha256", "changed"),
                           ("reranker_model", "different")):
            candidate = copy.deepcopy(baseline)
            candidate["config"][key] = value
            with self.subTest(key=key):
                self.assertTrue(any(key in issue for issue in compatibility_issues(baseline, candidate)))
        candidate = copy.deepcopy(baseline)
        del candidate["config"]["metric_version"]
        self.assertTrue(any("missing" in issue for issue in compatibility_issues(baseline, candidate)))
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["id"] = "different_case"
        self.assertIn("case IDs: evaluated question sets differ", compatibility_issues(baseline, candidate))

    def test_load_full_and_summary_reports_and_reject_corrupt_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = make_report()
            path.write_text(json.dumps(report))
            self.assertEqual(load_report(path), report)
            del report["cases"]
            path.write_text(json.dumps(report))
            self.assertEqual(load_report(path), report)
            self.assertIn("summary data only", compare_reports(report, report))
            for value in (float("nan"), -0.1, 1.5, "0.7", True):
                invalid = copy.deepcopy(report)
                invalid["overall"]["evidence"]["hit_at_3"] = value
                path.write_text(json.dumps(invalid))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_report(path)
            invalid = make_report()
            invalid["cases"][1]["id"] = invalid["cases"][0]["id"]
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_report(path)

    def test_cli_multiple_candidates_output_and_mismatch_override(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.json" for index in range(3)]
            reports = [make_report(), make_report((True, True), "intfloat/e5-small-v2"),
                       make_report((False, True), "BAAI/bge-small-en-v1.5")]
            for path, report in zip(paths, reports):
                path.write_text(json.dumps(report))
            output_path = Path(directory) / "comparison.txt"
            command = [sys.executable, str(ROOT / "compare_runs.py"), *map(str, paths)]
            result = subprocess.run(command + ["--output", str(output_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output_path.read_text(), result.stdout)
            self.assertEqual(result.stdout.count("Candidate:"), 2)
            self.assertIn("Ranking by Evidence Hit@3", result.stdout)
            reports[1]["config"]["fuzzy_threshold"] = .85
            paths[1].write_text(json.dumps(reports[1]))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fuzzy_threshold", result.stderr)
            self.assertNotIn("Evidence Hit@3:", result.stdout)
            result = subprocess.run(command + ["--allow-mismatch"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("EXPLORATORY", result.stdout)
            self.assertNotIn("Ranking by Evidence Hit@3", result.stdout)


class EmbeddingConfigurationTests(unittest.TestCase):
    def test_model_prefixes_and_overrides(self):
        self.assertEqual(embedding_prefixes("all-MiniLM-L6-v2"), ("", ""))
        for model in ("intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/multilingual-e5-base"):
            self.assertEqual(embedding_prefixes(model), ("query: ", "passage: "))
        self.assertEqual(embedding_prefixes("BAAI/bge-small-en-v1.5"),
                         ("Represent this sentence for searching relevant passages: ", ""))
        self.assertEqual(embedding_prefixes("other/model", "custom: ", "doc: "), ("custom: ", "doc: "))
        self.assertEqual(embedding_prefixes("intfloat/e5-small-v2", "", ""), ("", ""))
        instruct_model = "intfloat/multilingual-e5-large-instruct"
        instruction = (
            "Instruct: Given a question about scientific papers, "
            "retrieve relevant passages that answer the question\nQuery: "
        )
        self.assertEqual(embedding_prefixes(instruct_model), (instruction, ""))
        self.assertEqual(embedding_prefixes(instruct_model, query_prefix="custom: "),
                         ("custom: ", ""))
        self.assertEqual(embedding_prefixes(instruct_model, passage_prefix="doc: "),
                         (instruction, "doc: "))
        self.assertEqual(embedding_prefixes(instruct_model, "custom: ", "doc: "),
                         ("custom: ", "doc: "))
        self.assertEqual(embedding_prefixes(instruct_model, "", ""), ("", ""))

    def test_cache_signature_preserves_baseline_and_isolates_prefixes(self):
        docs = {"paper.pdf": {"pages": [{"page": 0, "text": "hello"}]}}
        baseline = embedding_signature(docs, EMBEDDING_MODEL_NAME, "", "")
        self.assertEqual(baseline, create_signature(docs, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL_NAME))
        self.assertNotEqual(baseline, embedding_signature(docs, "intfloat/e5-small-v2", "query: ", "passage: "))
        self.assertNotEqual(embedding_signature(docs, "model", "q: ", "p: "),
                            embedding_signature(docs, "model", "q: ", "different: "))

    def test_encoder_formatting_preserves_raw_chunks(self):
        class FakeModel:
            def encode(self, texts, **kwargs):
                self.texts, self.kwargs = texts, kwargs
                return [[1, 2, 3]]
        model = FakeModel()
        chunks = [{"text": "Original passage"}]
        self.assertEqual(encode_passages(model, chunks, "passage: "), [[1, 2, 3]])
        self.assertEqual(model.texts, ["passage: Original passage"])
        self.assertEqual(chunks[0]["text"], "Original passage")
        self.assertEqual(QueryEncoder(model, "query: ").encode(["Question?"], normalize_embeddings=True), [[1, 2, 3]])
        self.assertEqual(model.texts, ["query: Question?"])
        self.assertTrue(model.kwargs["normalize_embeddings"])
        self.assertEqual(model.kwargs["prompt"], "")

        query_prefix, passage_prefix = embedding_prefixes("intfloat/multilingual-e5-large-instruct")
        encoder = QueryEncoder(model, query_prefix)
        questions = ["What is retrieval?"]
        for _ in range(2):
            encoder.encode(questions, normalize_embeddings=True)
            self.assertEqual(model.texts, [query_prefix + questions[0]])
            self.assertEqual(model.texts[0].count("Instruct:"), 1)
            self.assertEqual(model.texts[0].count("Query:"), 1)
            self.assertEqual(model.kwargs["prompt"], "")
        self.assertEqual(questions, ["What is retrieval?"])
        encode_passages(model, chunks, passage_prefix)
        self.assertEqual(model.texts, ["Original passage"])
        self.assertEqual(chunks[0]["text"], "Original passage")
        self.assertEqual(model.kwargs["prompt"], "")


if __name__ == "__main__":
    unittest.main()
