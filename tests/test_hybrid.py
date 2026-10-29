import math
import unittest

import numpy as np

from src.lexical import BM25Index, tokenize
from src.retriever import reciprocal_rank_fusion, retrieve_hybrid, rerank


def chunk(index, text="text", document="paper.pdf"):
    return {"document": document, "page": 1, "chunk_id": index, "text": text}


class HybridTests(unittest.TestCase):
    def test_bm25_formula_and_rare_terms(self):
        chunks = [chunk(0, "retrieval retrieval colbert"), chunk(1, "retrieval training training")]
        index = BM25Index(chunks)
        results = index.retrieve("COLBERT retrieval retrieval")
        expected = math.log(2) + math.log(1.2) * 2 * 2.5 / 3.5
        self.assertEqual(results[0]["chunk_id"], 0)
        self.assertAlmostEqual(results[0]["bm25_score"], expected)
        self.assertEqual(index.retrieve("absent"), [])
        self.assertEqual(BM25Index([]).retrieve("colbert"), [])
        self.assertEqual(BM25Index([chunk(0, "")]).retrieve("x"), [])

    def test_tokenization_pdf_text(self):
        self.assertEqual(tokenize("COLBERT efﬁcient re-\ntrieval 61.9"),
                         ["colbert", "efficient", "retrieval", "61.9"])

    def test_length_normalization_and_empty_query(self):
        index = BM25Index([chunk(0, "colbert"), chunk(1, "colbert " + "filler " * 20)])
        self.assertEqual(index.retrieve("colbert")[0]["chunk_id"], 0)
        self.assertEqual(index.retrieve(""), [])
        self.assertEqual(index.retrieve("colbert", top_k=0), [])

    def test_rrf_overlap_scores_source_identity_and_no_mutation(self):
        dense = [{**chunk(0), "score": .9}, {**chunk(1), "score": .7}]
        lexical = [{**chunk(1), "bm25_score": 2}, chunk(0, document="other.pdf")]
        results = reciprocal_rank_fusion(dense, lexical)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["chunk_id"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 1/62 + 1/61)
        self.assertEqual(results[0]["score"], .7)
        self.assertEqual(results[0]["bm25_score"], 2)
        self.assertEqual(results[0]["dense_rank"], 2)
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertNotIn("rrf_score", dense[1])
        self.assertEqual(reciprocal_rank_fusion([], []), [])
        self.assertEqual(len(reciprocal_rank_fusion(dense, lexical, top_k=1)), 1)
        self.assertEqual(len(reciprocal_rank_fusion([dense[0]] * 2, [])), 1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_keyword_branch_recovers_chunk_missed_by_dense(self):
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.]])
        chunks = [chunk(0, "general background"), chunk(1, "ColBERT late interaction")]
        embeddings = np.array([[1., 0.], [0., 1.]])
        results = retrieve_hybrid("ColBERT", Encoder(), chunks, embeddings,
                                  BM25Index(chunks), top_k=2, branch_top_k=1)
        self.assertEqual({r["chunk_id"] for r in results}, {0, 1})
        keyword = next(r for r in results if r["chunk_id"] == 1)
        self.assertNotIn("score", keyword)
        self.assertEqual(keyword["bm25_rank"], 1)

    def test_empty_candidates_do_not_call_reranker(self):
        self.assertEqual(rerank("q", [], None), [])


if __name__ == "__main__":
    unittest.main()
