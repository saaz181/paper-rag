import unittest

from src.retriever import build_context


class BuildContextTests(unittest.TestCase):
    def test_distinct_chunks_on_same_page_keep_evidence_and_rank_order(self):
        results = [
            {"document": "paper.pdf", "page": 6, "chunk_id": 0,
             "text": "Background information."},
            {"document": "paper.pdf", "page": 6, "chunk_id": 1,
             "text": "The evidence needed to answer the question."},
        ]

        context = build_context(results)

        self.assertEqual(context, (
            "SOURCE: [paper.pdf, p. 6]\n\nBackground information."
            "\n\n---\n\n"
            "SOURCE: [paper.pdf, p. 6]\n\n"
            "The evidence needed to answer the question."
        ))

    def test_repeated_chunk_appears_once(self):
        chunk = {"document": "paper.pdf", "page": 6, "chunk_id": 1,
                 "text": "Supporting evidence."}

        context = build_context([chunk, dict(chunk)])

        self.assertEqual(context, "SOURCE: [paper.pdf, p. 6]\n\nSupporting evidence.")

    def test_identical_text_from_different_sources_keeps_citations(self):
        results = [
            {"document": "paper.pdf", "page": 6, "text": "Shared text."},
            {"document": "paper.pdf", "page": 7, "text": "Shared text."},
            {"document": "other.pdf", "page": 6, "text": "Shared text."},
        ]

        context = build_context(results)

        self.assertEqual(context.count("Shared text."), 3)
        for source in ("paper.pdf, p. 6", "paper.pdf, p. 7", "other.pdf, p. 6"):
            self.assertIn(f"SOURCE: [{source}]", context)

    def test_empty_results_produce_empty_context(self):
        self.assertEqual(build_context([]), "")


if __name__ == "__main__":
    unittest.main()
