"""Validate benchmark provenance and evidence against local PDFs, without models.

Run from the project root: python -m benchmarks.validate
"""
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from benchmarks.test_cases import load_benchmark, load_test_cases


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "benchmarks" / "manifest.json").read_text())
    cases = load_benchmark()
    pages = {}
    for doc in manifest["documents"]:
        path = root / "data" / doc["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == doc["sha256"], path
        pages[doc["filename"]] = [page.extract_text() or "" for page in PdfReader(path).pages]
        assert len(pages[doc["filename"]]) == doc["pages"]
        assert sum(c["expected_document"] == doc["filename"] for c in cases) == doc["case_count"]
    assert len(cases) == manifest["question_count"]
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({normalize(c["question"]) for c in cases}) == len(cases)
    assert dict(Counter(c["category"] for c in cases)) == manifest["category_counts"]
    for case in cases:
        assert case["question"].endswith("?"), case["id"]
        assert case["reference_answer"].strip(), case["id"]
        assert case["expected_page"] == case["pdf_page"] - 1, case["id"]
        text = pages[case["expected_document"]][case["expected_page"]]
        ev = case["evidence"]
        assert 0 <= ev["start_char"] < ev["end_char"] <= len(text), case["id"]
        assert text[ev["start_char"]:ev["end_char"]] == ev["quote"], case["id"]
        assert normalize(ev["anchor"]) in normalize(ev["quote"]), case["id"]
    simple = load_test_cases()
    one_based = load_test_cases(page_base=1)
    assert len(simple) == len(cases)
    for a, b in zip(simple, one_based):
        assert set(a) == {"question", "expected_document", "expected_page"}
        assert a["expected_document"] == b["expected_document"]
        assert a["expected_page"] + 1 == b["expected_page"]
    print(f"Validated {len(cases)} unique questions across {len(pages)} PDFs.")
    print("PDF hashes, page references, exact evidence spans, categories, and both page bases passed.")
    print("These checks verify provenance and structure, not semantic correctness of reference answers.")


if __name__ == "__main__":
    main()
