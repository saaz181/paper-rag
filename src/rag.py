import os
import json
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from src.chunking import chunk_docs
from src.embeddings import (
    load_embeddings,
    save_embeddings
)
from src.retriever import (
    retrieve,
    retrieve_hybrid,
    rerank,
    build_context
)
from src.generator import Generator
from src.lexical import BM25Index
from src.encoding import (
    QueryEncoder, embedding_cache_dir, embedding_prefixes, embedding_signature, encode_passages,
)
from src.retrieval_config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, RETRIEVAL_METHOD,
    CHUNK_SIZE, OVERLAP, RETRIEVAL_TOP_K, RERANK_TOP_K, BRANCH_TOP_K, RRF_K,
)
from src.cache import (
    save_chunks,
    load_chunks,
    save_cache_metadata,
    is_cache_valid
)


load_dotenv()

parser = argparse.ArgumentParser(description="Answer paper questions with dense or hybrid retrieval")
parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
parser.add_argument("--retrieval-method", choices=("dense", "hybrid"), default=RETRIEVAL_METHOD)
parser.add_argument("--query", help="Answer one question instead of the built-in examples")
parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME)
parser.add_argument("--rerank-top-k", type=int, choices=(3, 5), default=RERANK_TOP_K)
parser.add_argument("--branch-top-k", type=int, default=BRANCH_TOP_K)
parser.add_argument("--rrf-k", type=int, default=RRF_K)
parser.add_argument("--reranker-cache-dir", help="Optional Hugging Face cache for the reranker")
args = parser.parse_args()
if args.branch_top_k <= 0 or args.rrf_k <= 0:
    parser.error("--branch-top-k and --rrf-k must be positive")
if args.reranker_cache_dir is None and args.reranker_model == "BAAI/bge-reranker-base":
    if os.path.isdir("cache/models/models--BAAI--bge-reranker-base"):
        args.reranker_cache_dir = "cache/models"

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_JSON = "data.json"
CACHE_DIR = "cache"
EMBEDDING_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = args.embedding_model
RERANKER_MODEL_NAME = (
    args.reranker_model
)
GENERATOR_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)
RERANK_TOP_K = args.rerank_top_k



os.makedirs(CACHE_DIR, exist_ok=True)



with open(DATA_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)



query_prefix, passage_prefix = embedding_prefixes(EMBEDDING_MODEL_NAME)
current_signature = embedding_signature(documents, EMBEDDING_MODEL_NAME, query_prefix, passage_prefix)
CACHE_DIR = embedding_cache_dir(current_signature, EMBEDDING_MODEL_NAME, query_prefix, passage_prefix)
EMBEDDING_PATH = CACHE_DIR / "embeddings.npy"
CHUNKS_PATH = CACHE_DIR / "chunks.json"
METADATA_PATH = CACHE_DIR / "metadata.json"


cache_valid = is_cache_valid(
    metadata_path=METADATA_PATH,
    chunks_path=CHUNKS_PATH,
    embeddings_path=EMBEDDING_PATH,
    current_signature=current_signature
)



print(
    "\n"
    + "=" * 30
    + " Loading Embedding Model "
    + "=" * 30
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(
    "-" * 30
    + " Embedding Model Loaded "
    + "-" * 30
)



if cache_valid:

    print("\nLoading cached chunks and embeddings...")

    chunks = load_chunks(CHUNKS_PATH)

    embeddings = load_embeddings(EMBEDDING_PATH)

else:

    print("\nCache is missing or outdated.")

    print("Rebuilding chunks and embeddings...")

    chunks = chunk_docs(
        documents,
        max_words=CHUNK_SIZE,
        overlap=OVERLAP
    )

    embeddings = encode_passages(embedding_model, chunks, passage_prefix)

    save_chunks(chunks, CHUNKS_PATH)

    save_embeddings(embeddings, EMBEDDING_PATH)

    save_cache_metadata(current_signature, METADATA_PATH)

    print("Cache rebuilt successfully.")


print("\nNumber of chunks:", len(chunks))

print("Embedding shape:", embeddings.shape)
query_model = QueryEncoder(embedding_model, query_prefix)
lexical_index = BM25Index(chunks) if args.retrieval_method == "hybrid" else None




print(
    "\n"
    + "=" * 30
    + " Loading Reranker "
    + "=" * 30
)

reranker = CrossEncoder(RERANKER_MODEL_NAME,
                       **({"cache_folder": args.reranker_cache_dir}
                          if args.reranker_cache_dir else {}))

print(
    "-" * 30
    + " Reranker Loaded "
    + "-" * 30
)


print(
    "\n"
    + "=" * 30
    + " Loading Generator "
    + "=" * 30
)

generator = Generator(model_name=GENERATOR_MODEL_NAME)

print(
    "-" * 30
    + " Generator Loaded "
    + "-" * 30
)




queries = [
    "What is retrieval-augmented generation?",
    "How does RAG combine retrieval and generation?",
    (
        "What is the difference between parametric "
        "and non-parametric memory in RAG?"
    )
]
if args.query:
    queries = [args.query]




for query in queries:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "QUERY:",
        query
    )

    print(
        "=" * 80
    )


    if lexical_index is not None:
        candidates = retrieve_hybrid(
            query=query, model=query_model, chunks=chunks,
            embeddings=embeddings, lexical_index=lexical_index,
            top_k=RETRIEVAL_TOP_K, branch_top_k=args.branch_top_k, rrf_k=args.rrf_k)
    else:
        candidates = retrieve(
            query=query,
            model=query_model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=RETRIEVAL_TOP_K,
            min_score=None
        )

    results = rerank(
        query=query,
        candidates=candidates,
        reranker=reranker,
        top_k=RERANK_TOP_K
    )

    context = build_context(results)

    answer = generator.generate(
        query=query,
        context=context
    )

    print("\nANSWER:")
    print(answer)


    print("\nSOURCES:")

    seen_sources = set()

    for result in results:

        source = (
            result["document"],
            result["page"]
        )

        if source in seen_sources:
            continue

        seen_sources.add(source)

        print(
            f"- {result['document']}, "
            f"page {result['page']}"
        )
