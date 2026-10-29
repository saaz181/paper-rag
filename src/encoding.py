import json
from pathlib import Path

from src.cache import create_signature
from src.retrieval_config import CHUNK_SIZE, OVERLAP

def embedding_prefixes(model_name, query_prefix=None, passage_prefix=None):
    e5 = {f"intfloat/e5-{size}-v2" for size in ("small", "base", "large")}

    e5 |= {f"intfloat/multilingual-e5-{size}" for size in ("small", "base", "large")}

    bge = {f"BAAI/bge-{size}-en{version}" for size in ("small", "base", "large")
           for version in ("", "-v1.5")}

    defaults = ("", "")

    if model_name in e5:
        defaults = ("query: ", "passage: ")

    elif model_name in bge:
        defaults = ("Represent this sentence for searching relevant passages: ", "")

    elif "e5" in model_name.lower() and "instruct" in model_name.lower():
        defaults = (
            "Instruct: Given a question about scientific papers, "
            "retrieve relevant passages that answer the question\n"
            "Query: ",
            ""
        )
    return (defaults[0] if query_prefix is None else query_prefix,
            defaults[1] if passage_prefix is None else passage_prefix)


def embedding_signature(documents, model_name, query_prefix, passage_prefix):
    key = model_name if not (query_prefix or passage_prefix) else json.dumps(
        {"model": model_name, "query_prefix": query_prefix, "passage_prefix": passage_prefix},
        sort_keys=True)
    return create_signature(documents, CHUNK_SIZE, OVERLAP, key)


def encode_passages(model, chunks, prefix):
    return model.encode([prefix + chunk["text"] for chunk in chunks],
                        prompt="", normalize_embeddings=True, show_progress_bar=True)


class QueryEncoder:
    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix

    def encode(self, texts, **kwargs):
        return self.model.encode([self.prefix + text for text in texts], prompt="", **kwargs)


def embedding_cache_dir(signature, model_name, query_prefix, passage_prefix, cache_root="cache"):
    root = Path(cache_root)
    if model_name == "all-MiniLM-L6-v2" and not (query_prefix or passage_prefix):
        return root
    return root / "experiments" / signature
