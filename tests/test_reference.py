import json
from pathlib import Path
import unittest

from src.encoding import embedding_cache_dir, embedding_prefixes, embedding_signature
from src import retrieval_config as reference


class ReferenceTests(unittest.TestCase):
    def test_defaults_match_measured_reference_snapshot(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/baselines/e5_instruct_bm25_minilm_reference.json"
        snapshot = json.loads(path.read_text())
        config = snapshot["config"]
        self.assertEqual(config["embedding_model"], reference.EMBEDDING_MODEL_NAME)
        self.assertEqual(config["reranker_model"], reference.RERANKER_MODEL_NAME)
        self.assertEqual(config["retrieval_method"], reference.RETRIEVAL_METHOD)
        self.assertEqual(config["rerank_top_k"], reference.RERANK_TOP_K)
        self.assertEqual(config["retrieval_top_k"], reference.RETRIEVAL_TOP_K)
        self.assertEqual(config["hybrid_config"]["branch_top_k"], reference.BRANCH_TOP_K)
        self.assertEqual(config["hybrid_config"]["rrf_k"], reference.RRF_K)
        self.assertEqual(embedding_prefixes(reference.EMBEDDING_MODEL_NAME),
                         (config["query_prefix"], config["passage_prefix"]))
        experiment = snapshot["top5_experiment"]
        self.assertEqual(experiment["config"]["rerank_top_k"], 5)
        self.assertEqual(config["chunks_sha256"], experiment["config"]["chunks_sha256"])
        self.assertEqual(config["benchmark_sha256"], experiment["config"]["benchmark_sha256"])

    def test_reference_encoding_uses_isolated_cache_and_preserves_legacy(self):
        docs = {"p": {"pages": [{"page": 0, "text": "passage"}]}}
        query, passage = embedding_prefixes(reference.EMBEDDING_MODEL_NAME)
        signature = embedding_signature(docs, reference.EMBEDDING_MODEL_NAME, query, passage)
        self.assertEqual(embedding_cache_dir(signature, reference.EMBEDDING_MODEL_NAME, query, passage),
                         Path("cache/experiments") / signature)
        self.assertNotEqual(signature, embedding_signature(docs, reference.EMBEDDING_MODEL_NAME, "", ""))
        self.assertEqual(embedding_cache_dir("old", "all-MiniLM-L6-v2", "", ""), Path("cache"))
        self.assertEqual(embedding_cache_dir("prefixed", "all-MiniLM-L6-v2", "custom: ", ""),
                         Path("cache/experiments/prefixed"))


if __name__ == "__main__":
    unittest.main()
