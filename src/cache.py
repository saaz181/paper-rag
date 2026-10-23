import os
import json
import hashlib


def create_signature(
    documents,
    chunk_size,
    overlap,
    embedding_model
):
    config = {
        "documents": documents,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model
    }

    raw = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def save_chunks(chunks, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )


def load_chunks(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_cache_metadata(path):
    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_cache_metadata(
    signature,
    path
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "signature": signature
            },
            f,
            indent=2
        )


def is_cache_valid(
    metadata_path,
    chunks_path,
    embeddings_path,
    current_signature
):
    metadata = load_cache_metadata(
        metadata_path
    )

    return (
        metadata is not None
        and metadata.get("signature")
        == current_signature
        and os.path.exists(chunks_path)
        and os.path.exists(embeddings_path)
    )