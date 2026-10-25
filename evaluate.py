"""Evaluate retrieval with comparable page, exact-anchor and fuzzy-anchor scores."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from benchmarks.metrics import (
    FUZZY_THRESHOLD,
    evidence_matches,
    exact_evidence_matches,
    hit_at_k,
    normalize_text,
    page_matches,
    reciprocal_rank,
    strict_evidence_matches,
)
from benchmarks.test_cases import load_benchmark
from src.cache import (
    create_signature, is_cache_valid, load_chunks, save_cache_metadata, save_chunks,
)
from src.chunking import chunk_docs

DATA_JSON = "data.json"
CACHE_DIR = "cache"
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CHUNK_SIZE = 150
OVERLAP = 25
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 3
FAILED_CASES_PATH = "evaluation_failures.json"
REPORT_PATH = "evaluation_results.json"
FAILURE_TYPES = ("chunking_failure", "retrieval_failure", "reranking_failure")


def detect_page_base(documents):
    """Infer physical page numbering from the full ingested documents."""
    bases = {min(page["page"] for page in doc["pages"])
             for doc in documents.values() if doc["pages"]}
    if len(bases) != 1 or not bases.issubset({0, 1}):
        raise ValueError("Documents must consistently use 0-based or 1-based pages")
    return bases.pop()


def validate_cases(cases, documents):
    """Reject incomplete labels before loading expensive models."""
    if not cases:
        raise ValueError("No benchmark cases to evaluate")
    pages = {name: {page["page"] for page in doc["pages"]}
             for name, doc in documents.items()}
    for case in cases:
        label = case.get("id", case.get("question", "<unknown>"))
        if not normalize_text(case.get("evidence", {}).get("anchor", "")):
            raise ValueError(f"{label}: missing evidence anchor; use load_benchmark()")
        if not case.get("category"):
            raise ValueError(f"{label}: missing category; use load_benchmark()")
        if case["expected_page"] not in pages.get(case["expected_document"], set()):
            raise ValueError(f"{label}: expected document/page is absent from data.json")


def classify_failure(case, document_chunks, candidates, results,
                     match_function=evidence_matches):
    """Identify the first stage that lost the evidence under the chosen matcher.

    Call with all chunks from the expected document for document-wide evidence;
    stored chunk audits only describe the old designated-page anchor metric.
    """
    if not any(match_function(chunk, case) for chunk in document_chunks):
        return "chunking_failure"
    if not any(match_function(candidate, case) for candidate in candidates):
        return "retrieval_failure"
    if not any(match_function(result, case) for result in results):
        return "reranking_failure"
    return None


def matchers(threshold=FUZZY_THRESHOLD):
    return {
        "page": page_matches,
        "strict_evidence": strict_evidence_matches,
        "exact_evidence": exact_evidence_matches,
        "evidence": lambda result, case: evidence_matches(result, case, threshold),
    }


def analyze_case(case, document_chunks, candidates, results, threshold=FUZZY_THRESHOLD):
    """Score one case independently of model loading, using full chunk text."""
    methods = matchers(threshold)
    row = {key: case[key] for key in
           ("id", "question", "category", "expected_document", "expected_page")}
    row["evidence_anchor"] = case["evidence"]["anchor"]
    for name, matcher in methods.items():
        for k in (1, 3):
            row[f"{name}_hit_{k}"] = bool(hit_at_k(results, case, k, matcher))
        row[f"{name}_rr"] = reciprocal_rank(results, case, matcher)
    row["failure_types"] = {
        name: classify_failure(case, document_chunks, candidates, results, methods[name])
        for name in ("strict_evidence", "exact_evidence", "evidence")
    }
    row["failure_type"] = row["failure_types"]["evidence"]
    row["retrieved_results"] = [
        {
            "rank": rank,
            "document": result["document"],
            "page": result["page"],
            "chunk_id": result.get("chunk_id"),
            "embedding_score": result.get("score"),
            "rerank_score": result.get("rerank_score"),
            "matches": {name: matcher(result, case) for name, matcher in methods.items()},
            # Keep complete text so a reviewer can reproduce the match decision.
            "text": result["text"],
        }
        for rank, result in enumerate(results, start=1)
    ]
    return row


def summarize(rows):
    total = len(rows)
    if not total:
        raise ValueError("Cannot summarize an empty evaluation")
    summary = {"total": total}
    for name in matchers():
        summary[name] = {
            "hit_at_1": sum(row[f"{name}_hit_1"] for row in rows) / total,
            "hit_at_3": sum(row[f"{name}_hit_3"] for row in rows) / total,
            "mrr": sum(row[f"{name}_rr"] for row in rows) / total,
        }
    summary["failure_counts"] = {
        name: {failure: sum(row["failure_types"][name] == failure for row in rows)
               for failure in FAILURE_TYPES}
        for name in ("strict_evidence", "exact_evidence", "evidence")
    }
    return summary


def print_metrics(summary):
    for key, label in (
        ("page", "Page"), ("strict_evidence", "Strict page evidence (previous metric)"),
        ("exact_evidence", "Exact document evidence"), ("evidence", "Evidence (fuzzy document)"),
    ):
        scores = summary[key]
        print(f"\n{label}")
        print(f"Hit@1: {scores['hit_at_1']:.3f}")
        print(f"Hit@3: {scores['hit_at_3']:.3f}")
        print(f"MRR:   {scores['mrr']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD)
    parser.add_argument("--report", default=REPORT_PATH, help="JSON summary and all case results")
    args = parser.parse_args()
    if not 0 < args.fuzzy_threshold <= 1:
        parser.error("--fuzzy-threshold must be greater than 0 and at most 1")

    with open(DATA_JSON, encoding="utf-8") as stream:
        documents = json.load(stream)
    page_base = detect_page_base(documents)
    cases = load_benchmark(page_base=page_base)
    validate_cases(cases, documents)
    print(f"Loaded {len(cases)} full benchmark cases (page base: {page_base}).", flush=True)

    from dotenv import load_dotenv
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from src.embeddings import embd_chunks, load_embeddings, save_embeddings
    from src.retriever import retrieve, rerank

    load_dotenv()
    signature = create_signature(documents, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL_NAME)
    cache_valid = is_cache_valid(METADATA_PATH, CHUNKS_PATH, EMBEDDINGS_PATH, signature)
    if cache_valid:
        print("Loading cached chunks and embeddings...")
        chunks = load_chunks(CHUNKS_PATH)
        embeddings = load_embeddings(EMBEDDINGS_PATH)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    else:
        print("Rebuilding chunks and embeddings...")
        chunks = chunk_docs(documents, max_words=CHUNK_SIZE, overlap=OVERLAP)
        embedding_model, embeddings = embd_chunks(chunks, model_name=EMBEDDING_MODEL_NAME)
        save_chunks(chunks, CHUNKS_PATH)
        save_embeddings(embeddings, EMBEDDINGS_PATH)
        save_cache_metadata(signature, METADATA_PATH)
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    chunks_by_document = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document"]].append(chunk)
    rows = []
    for index, case in enumerate(cases, start=1):
        candidates = retrieve(query=case["question"], model=embedding_model,
                              chunks=chunks, embeddings=embeddings,
                              top_k=RETRIEVAL_TOP_K, min_score=None)
        results = rerank(query=case["question"], candidates=candidates,
                         reranker=reranker, top_k=RERANK_TOP_K)
        rows.append(analyze_case(case, chunks_by_document[case["expected_document"]],
                                 candidates, results, args.fuzzy_threshold))
        if index % 25 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)}", flush=True)

    overall = summarize(rows)
    category_rows = defaultdict(list)
    for row in rows:
        category_rows[row["category"]].append(row)
    categories = {category: summarize(group) for category, group in sorted(category_rows.items())}
    print("\n" + "=" * 70 + "\nFINAL RETRIEVAL EVALUATION\n" + "=" * 70)
    print(f"Test questions: {len(cases)}")
    print(f"Fuzzy threshold: {args.fuzzy_threshold:.2f}")
    print_metrics(overall)
    print("\nPER-CATEGORY RESULTS")
    for category, summary in categories.items():
        print(f"\nCategory: {category}\nN: {summary['total']}")
        print_metrics(summary)
    print("\nFAILURE ANALYSIS (fuzzy document evidence)")
    for failure, count in overall["failure_counts"]["evidence"].items():
        print(f"{failure}: {count} ({count / len(cases):.1%})")
    failed = [row for row in rows if row["failure_type"]]
    print(f"\nTotal failed cases: {len(failed)}")

    report = {
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME, "reranker_model": RERANKER_MODEL_NAME,
            "chunk_size_words": CHUNK_SIZE, "overlap_words": OVERLAP,
            "retrieval_top_k": RETRIEVAL_TOP_K, "rerank_top_k": RERANK_TOP_K,
            "page_base": page_base, "corpus_signature": signature,
            "chunks_sha256": hashlib.sha256(Path(CHUNKS_PATH).read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path("benchmarks/cases.json").read_bytes()).hexdigest(),
            "metric_version": "document_fuzzy_v1", "fuzzy_threshold": args.fuzzy_threshold,
            "numeric_fuzzy_guard": True, "fuzzy_window_word_tolerance": 1,
            "evidence_scope": "expected_document_all_pages",
            "normalization": "nfkc_casefold_pdf_hyphenation_punctuation_whitespace",
            "mrr_cutoff": RERANK_TOP_K,
        },
        "overall": overall, "categories": categories, "cases": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(FAILED_CASES_PATH).write_text(json.dumps(failed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Failure details saved to: {FAILED_CASES_PATH}")
    print(f"Full evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
