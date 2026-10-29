"""Evaluate retrieval with comparable page, exact-anchor and fuzzy-anchor scores."""
import argparse
import hashlib
import json
import os
import platform
from time import perf_counter
from collections import defaultdict
from pathlib import Path

from benchmarks.metrics import (
    FUZZY_THRESHOLD,
    METRIC_VERSION,
    NORMALIZATION,
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
from src.encoding import (
    QueryEncoder, embedding_cache_dir, embedding_prefixes, embedding_signature, encode_passages,
)
from src.retrieval_config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, RETRIEVAL_METHOD, CHUNK_SIZE, OVERLAP,
    RETRIEVAL_TOP_K, RERANK_TOP_K, BRANCH_TOP_K, RRF_K,
)

DATA_JSON = "data.json"
CACHE_DIR = "cache"
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")
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


def analyze_case(case, document_chunks, candidates, results, threshold=FUZZY_THRESHOLD,
                 final_top_k=3):
    """Score one case independently of model loading, using full chunk text."""
    results = results[:final_top_k]
    methods = matchers(threshold)
    row = {key: case[key] for key in
           ("id", "question", "category", "expected_document", "expected_page")}

    row["evidence_anchor"] = case["evidence"]["anchor"]
    row["evaluation_cutoff"] = final_top_k

    for name, matcher in methods.items():
        for k in (k for k in (1, 3, 5) if k <= final_top_k):
            row[f"{name}_hit_{k}"] = bool(hit_at_k(results, case, k, matcher))
        row[f"{name}_rr"] = reciprocal_rank(results[:final_top_k], case, matcher)
        for k in (k for k in (3, 5) if k <= final_top_k):
            row[f"{name}_rr_{k}"] = reciprocal_rank(results[:k], case, matcher)

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
            "embedding_score": result.get("score", result.get("embedding_score")),
            **{key: result[key] for key in
               ("bm25_score", "rrf_score", "dense_rank", "bm25_rank") if key in result},
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
            **{f"hit_at_{k}": sum(row[f"{name}_hit_{k}"] for row in rows) / total
               for k in (1, 3, 5) if all(f"{name}_hit_{k}" in row for row in rows)},
            **{f"mrr_at_{k}": sum(row[f"{name}_rr_{k}"] for row in rows) / total
               for k in (3, 5) if all(f"{name}_rr_{k}" in row for row in rows)},
            "mrr": sum(row[f"{name}_rr"] for row in rows) / total,
        }
    summary["failure_counts"] = {
        name: {failure: sum(row["failure_types"][name] == failure for row in rows)
               for failure in FAILURE_TYPES}
        for name in ("strict_evidence", "exact_evidence", "evidence")
    }
    for field in ("candidate_counts", "latency_ms"):
        if all(field in row for row in rows):
            summary[f"average_{field}"] = {
                key: sum(row[field][key] for row in rows) / total for key in rows[0][field]}
    return summary


def print_metrics(summary):
    for key, label in (
        ("page", "Page"), ("strict_evidence", "Strict page evidence (previous metric)"),
        ("exact_evidence", "Exact document evidence"), ("evidence", "Evidence (fuzzy document)"),
    ):
        scores = summary[key]
        print(f"\n{label}")
        print(f"Hit@1: {scores['hit_at_1']:.3f}")
        for k in (3, 5):
            if f"hit_at_{k}" in scores:
                print(f"Hit@{k}: {scores[f'hit_at_{k}']:.3f}")
        for k in (3, 5):
            if f"mrr_at_{k}" in scores:
                print(f"MRR@{k}: {scores[f'mrr_at_{k}']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME)
    parser.add_argument("--rerank-top-k", type=int, choices=(3, 5), default=RERANK_TOP_K)
    parser.add_argument("--reranker-cache-dir", type=Path,
                        help="Optional Hugging Face cache containing reranker model files")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"),
                        help="Run both models on a specified device; default is automatic")
    parser.add_argument("--retrieval-method", choices=("dense", "hybrid"), default=RETRIEVAL_METHOD)
    parser.add_argument("--branch-top-k", type=int, default=BRANCH_TOP_K,
                        help="Candidates per branch before hybrid fusion (final budget stays 20)")
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help="Positive RRF rank constant")
    parser.add_argument("--query-prefix", default=None, help="Override the model's query prefix")
    parser.add_argument("--passage-prefix", default=None, help="Override the model's passage prefix")
    parser.add_argument("--cache-dir", type=Path, help="Optional cache directory; model runs are isolated by default")
    parser.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD)
    parser.add_argument("--report", default=REPORT_PATH, help="JSON summary and all case results")
    args = parser.parse_args()
    if args.reranker_cache_dir is None and args.reranker_model == "BAAI/bge-reranker-base":
        if Path("cache/models/models--BAAI--bge-reranker-base").is_dir():
            args.reranker_cache_dir = Path("cache/models")
    if args.branch_top_k <= 0 or args.rrf_k <= 0:
        parser.error("--branch-top-k and --rrf-k must be positive")
    if args.retrieval_method == "hybrid" and args.report == REPORT_PATH:
        args.report = "runs/hybrid/evaluation_results.json"
    if not 0 < args.fuzzy_threshold <= 1:
        parser.error("--fuzzy-threshold must be greater than 0 and at most 1")

    try:
        query_prefix, passage_prefix = embedding_prefixes(
            args.embedding_model, args.query_prefix, args.passage_prefix)
    except ValueError as error:
        parser.error(str(error))

    with open(DATA_JSON, encoding="utf-8") as stream:
        documents = json.load(stream)

    page_base = detect_page_base(documents)
    cases = load_benchmark(page_base=page_base)

    validate_cases(cases, documents)
    print(f"Loaded {len(cases)} full benchmark cases (page base: {page_base}).", flush=True)

    from dotenv import load_dotenv
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from src.embeddings import load_embeddings, save_embeddings
    from src.retriever import retrieve, retrieve_hybrid, rerank
    from src.lexical import BM25Index

    load_dotenv()

    signature = embedding_signature(documents, args.embedding_model, query_prefix, passage_prefix)
    cache_dir = args.cache_dir or embedding_cache_dir(
        signature, args.embedding_model, query_prefix, passage_prefix, CACHE_DIR)
    chunks_path = cache_dir / "chunks.json"
    embeddings_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "metadata.json"
    cache_valid = is_cache_valid(metadata_path, chunks_path, embeddings_path, signature)
    print(f"Embedding model: {args.embedding_model}")
    print(f"Query prefix: {query_prefix!r}; passage prefix: {passage_prefix!r}")
    print(f"Cache directory: {cache_dir}")
    embedding_model = SentenceTransformer(args.embedding_model, device=args.device)
    if cache_valid:
        print("Loading cached chunks and embeddings...")
        chunks = load_chunks(chunks_path)
        embeddings = load_embeddings(embeddings_path)
    else:
        print("Rebuilding chunks and embeddings...")
        chunks = chunk_docs(documents, max_words=CHUNK_SIZE, overlap=OVERLAP)
        embeddings = encode_passages(embedding_model, chunks, passage_prefix)
        save_chunks(chunks, chunks_path)
        save_embeddings(embeddings, embeddings_path)
        save_cache_metadata(signature, metadata_path)
    query_model = QueryEncoder(embedding_model, query_prefix)
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    reranker = CrossEncoder(args.reranker_model, device=args.device,
                           **({"cache_folder": str(args.reranker_cache_dir)}
                              if args.reranker_cache_dir else {}))
    lexical_index = BM25Index(chunks) if args.retrieval_method == "hybrid" else None
    print(f"Retrieval method: {args.retrieval_method}")
    print(f"Reranker: {args.reranker_model}; final top {args.rerank_top_k}", flush=True)

    query_model.encode([cases[0]["question"]], normalize_embeddings=True)
    rerank(cases[0]["question"], [dict(chunk) for chunk in chunks[:RETRIEVAL_TOP_K]],
           reranker, top_k=args.rerank_top_k)

    chunks_by_document = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document"]].append(chunk)
    rows = []
    for index, case in enumerate(cases, start=1):
        start = perf_counter()
        if lexical_index is not None:
            candidates, diagnostics = retrieve_hybrid(
                query=case["question"], model=query_model, chunks=chunks,
                embeddings=embeddings, lexical_index=lexical_index,
                top_k=RETRIEVAL_TOP_K, branch_top_k=args.branch_top_k, rrf_k=args.rrf_k,
                return_diagnostics=True)
        else:
            candidates = retrieve(query=case["question"], model=query_model,
                                  chunks=chunks, embeddings=embeddings,
                                  top_k=RETRIEVAL_TOP_K, min_score=None)
            diagnostics = {"candidate_counts": {"dense": len(candidates), "bm25": 0,
                           "overlap": 0, "union": len(candidates), "reranker_input": len(candidates)},
                           "latency_ms": {"dense": (perf_counter() - start) * 1000,
                                          "bm25": 0.0, "fusion": 0.0}}

        retrieval_ms = (perf_counter() - start) * 1000
        rerank_start = perf_counter()
        results = rerank(query=case["question"], candidates=candidates,
                         reranker=reranker, top_k=args.rerank_top_k)
        rerank_ms = (perf_counter() - rerank_start) * 1000
        diagnostics["latency_ms"].update(retrieval=retrieval_ms, rerank=rerank_ms,
                                         total=(perf_counter() - start) * 1000)

        row = analyze_case(case, chunks_by_document[case["expected_document"]],
                           candidates, results, args.fuzzy_threshold, args.rerank_top_k)
        row.update(diagnostics)
        row["candidates"] = [{key: candidate[key] for key in
                              ("document", "page", "chunk_id", "score", "bm25_score",
                               "rrf_score", "dense_rank", "bm25_rank") if key in candidate}
                             for candidate in candidates]
        rows.append(row)
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
    print("Average candidate counts:", overall["average_candidate_counts"])
    print("Average latency (ms):", overall["average_latency_ms"])
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
            "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
            "retrieval_method": args.retrieval_method,
            "hybrid_config": ({"branch_top_k": args.branch_top_k, "rrf_k": args.rrf_k,
                               "bm25_k1": lexical_index.k1, "bm25_b": lexical_index.b,
                               "tokenizer": "nfkc_casefold_words_decimals_v1"}
                              if lexical_index is not None else None),
            "query_prefix": query_prefix, "passage_prefix": passage_prefix,
            "embedding_dimension": int(embeddings.shape[1]),
            "documents_sha256": hashlib.sha256(json.dumps(documents, sort_keys=True,
                ensure_ascii=False).encode("utf-8")).hexdigest(),
            "chunk_size_words": CHUNK_SIZE, "overlap_words": OVERLAP,
            "retrieval_top_k": RETRIEVAL_TOP_K, "rerank_top_k": args.rerank_top_k,
            "page_base": page_base, "corpus_signature": signature,
            "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path("benchmarks/cases.json").read_bytes()).hexdigest(),
            "metric_version": METRIC_VERSION, "fuzzy_threshold": args.fuzzy_threshold,
            "numeric_fuzzy_guard": True, "fuzzy_window_word_tolerance": 1,
            "evidence_scope": "expected_document_all_pages",
            "normalization": NORMALIZATION,
            "mrr_cutoff": args.rerank_top_k,
            "failure_cutoff": args.rerank_top_k,
            "latency_protocol": "single_query_sequential_warm_one_query_excludes_model_loading_and_metrics",
            "embedding_device": str(embedding_model.device),
            "reranker_device": str(reranker.device),
            "platform": platform.platform(),
        },
        "overall": overall, "categories": categories, "cases": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures_path = (Path(FAILED_CASES_PATH) if report_path == Path(REPORT_PATH)
                     else report_path.with_name(report_path.stem + "_failures.json"))
    failures_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Failure details saved to: {failures_path}")
    print(f"Full evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
