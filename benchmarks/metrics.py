"""Retrieval metrics: designated pages and lexical evidence proxies."""
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

FUZZY_THRESHOLD = 0.80


@lru_cache(maxsize=8192)
def normalize_text(text):
    """Normalize PDF ligatures, line-break hyphenation, punctuation and spaces."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


@lru_cache(maxsize=8192)
def normalize_evidence(text):
    """Legacy normalization, retained for comparison with the previous baseline."""
    return "".join(
        char for char in unicodedata.normalize("NFKC", text).casefold()
        if char.isalnum()
    )


def _exact_contains(anchor, text):
    # Token boundaries avoid matching '61' in '161' or 'late' in 'unrelated'.
    return bool(anchor) and f" {anchor} " in f" {text} "


@lru_cache(maxsize=8192)
def fuzzy_contains(anchor, text, threshold=FUZZY_THRESHOLD):
    """Compare short word windows using SequenceMatcher.

    Purely numeric anchors require an exact match. Other fuzzy matches must
    preserve all numeric tokens, to avoid rewarding incorrect factual numbers.
    This is a lexical heuristic, not a semantic evidence judge.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")
    anchor = normalize_text(anchor)
    text = normalize_text(text)
    if not anchor or not text:
        return False
    if _exact_contains(anchor, text):
        return True
    if not any(char.isalpha() for char in anchor):
        return False

    anchor_words = anchor.split()
    text_words = text.split()
    window_size = len(anchor_words)
    anchor_numbers = re.findall(r"\d+", anchor)
    # Allow one joined/split word, common with PDF line breaks and compounds.
    for size in range(max(1, window_size - 1), window_size + 2):
        for i in range(len(text_words) - size + 1):
            window = " ".join(text_words[i:i + size])
            if re.findall(r"\d+", window) != anchor_numbers:
                continue
            matcher = SequenceMatcher(None, anchor, window)
            if (matcher.real_quick_ratio() >= threshold
                    and matcher.quick_ratio() >= threshold
                    and matcher.ratio() >= threshold):
                return True
    return False


def page_matches(result, case):
    return (result["document"] == case["expected_document"]
            and result["page"] == case["expected_page"])


def strict_evidence_matches(result, case):
    """Previous baseline: designated page plus legacy normalized anchor."""
    if not page_matches(result, case):
        return False
    anchor = normalize_evidence(case.get("evidence", {}).get("anchor", ""))
    return bool(anchor) and anchor in normalize_evidence(result["text"])


def exact_evidence_matches(result, case):
    """Normalized exact anchor anywhere in the expected document."""
    if result["document"] != case["expected_document"]:
        return False
    anchor = normalize_text(case.get("evidence", {}).get("anchor", ""))
    return _exact_contains(anchor, normalize_text(result["text"]))


def evidence_matches(result, case, threshold=FUZZY_THRESHOLD):
    """Normalized fuzzy anchor anywhere in the expected document."""
    if result["document"] != case["expected_document"]:
        return False
    return fuzzy_contains(case.get("evidence", {}).get("anchor", ""),
                          result["text"], threshold=threshold)


def hit_at_k(results, case, k, match_function):
    return int(any(match_function(result, case) for result in results[:k]))


def reciprocal_rank(results, case, match_function):
    rank = find_first_rank(results, case, match_function)
    return 1 / rank if rank is not None else 0


def find_first_rank(results, case, match_function):
    return next((rank for rank, result in enumerate(results, start=1)
                 if match_function(result, case)), None)
