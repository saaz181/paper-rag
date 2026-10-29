import numpy as np
from time import perf_counter


def reciprocal_rank_fusion(dense_results, lexical_results, top_k=20, rrf_k=60):
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k <= 0:
        return []
    merged = {}
    for branch, results in (("dense", dense_results), ("bm25", lexical_results)):
        seen = set()
        for rank, result in enumerate(results, 1):
            key = (result["document"], result["page"], result["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, {**result, "rrf_score": 0.0})
            item.update({k: v for k, v in result.items() if k in ("score", "bm25_score")})
            item[f"{branch}_rank"] = rank
            item["rrf_score"] += 1 / (rrf_k + rank)
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]


def retrieve_hybrid(query, model, chunks, embeddings, lexical_index,
                    top_k=20, branch_top_k=20, rrf_k=60, return_diagnostics=False):
    start = perf_counter()
    dense = retrieve(query, model, chunks, embeddings, top_k=branch_top_k)
    dense_seconds = perf_counter() - start
    start = perf_counter()
    lexical = lexical_index.retrieve(query, top_k=branch_top_k)
    bm25_seconds = perf_counter() - start
    start = perf_counter()
    results = reciprocal_rank_fusion(dense, lexical, top_k=top_k, rrf_k=rrf_k)
    fusion_seconds = perf_counter() - start
    if not return_diagnostics:
        return results
    identity = lambda result: (result["document"], result["page"], result["chunk_id"])
    dense_ids, lexical_ids = set(map(identity, dense)), set(map(identity, lexical))
    return results, {
        "candidate_counts": {"dense": len(dense), "bm25": len(lexical),
                             "overlap": len(dense_ids & lexical_ids),
                             "union": len(dense_ids | lexical_ids), "reranker_input": len(results)},
        "latency_ms": {"dense": dense_seconds * 1000, "bm25": bm25_seconds * 1000,
                       "fusion": fusion_seconds * 1000},
    }


def retrieve(
    query,
    model,
    chunks,
    embeddings,
    top_k=5,
    min_score=None
):
    if top_k <= 0 or not chunks:
        return []
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = (query_embedding @ embeddings.T)[0]

    best_indices = np.argsort(scores)[::-1]

    results = []

    for idx in best_indices:
        score = float(scores[idx])

        if min_score is not None and score < min_score:
            continue

        result = {
            "score": score,
            "document": chunks[idx]["document"],
            "page": chunks[idx]["page"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"]
        }

        results.append(result)

        if len(results) == top_k:
            break

    return results


def rerank(query, candidates, reranker, top_k=3):
    if not candidates or top_k <= 0:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]



def build_context(results):
    context_parts = []

    seen = set()

    for result in results:
        source_key = (
            result["document"],
            result["page"],
            result["text"]
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        context_parts.append(
            f"""
SOURCE: [{result['document']}, p. {result['page']}]

{result['text']}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)
