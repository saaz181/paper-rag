import numpy as np

def retrieve(
    query,
    model,
    chunks,
    embeddings,
    top_k=5,
    min_score=None
):
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
            result["page"]
        )

        # Avoid duplicate document/page entries
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