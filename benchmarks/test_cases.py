"""Load the three-field retrieval cases without loading models or accessing the network."""
import json
from pathlib import Path


def load_benchmark(page_base=0):
    """Load reference answers and evidence; expected_page uses the requested base.

    pdf_page always uses physical 1-based PDF page numbers. Evidence offsets are
    offsets into pypdf's unmodified page text. Chunk audits describe the original
    zero-based data.json at max_words=150 and overlap=25.
    """
    if type(page_base) is not int or page_base not in (0, 1):
        raise ValueError("page_base must be 0 or 1")
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    for case in cases:
        case["expected_page"] = case["pdf_page"] - 1 + page_base
    return cases


def load_test_cases(page_base=0):
    """Return dictionaries compatible with evaluate.py's existing case format."""
    return [
        {key: case[key] for key in ("question", "expected_document", "expected_page")}
        for case in load_benchmark(page_base)
    ]


test_cases = load_test_cases()
