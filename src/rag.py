import os
import json

from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from chunking import chunk_docs
from embeddings import (
    embd_chunks,
    load_embeddings,
    save_embeddings
)
from retriever import (
    retrieve,
    rerank,
    build_context
)
from generator import Generator
from cache import (
    create_signature,
    save_chunks,
    load_chunks,
    save_cache_metadata,
    is_cache_valid
)


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_JSON = "data.json"
CACHE_DIR = "cache"
EMBEDDING_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)
GENERATOR_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)
CHUNK_SIZE = 150
OVERLAP = 25
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 3



# CREATE CACHE DIRECTORY
os.makedirs(CACHE_DIR, exist_ok=True)



# LOAD DOCUMENTS
with open(DATA_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)



# CREATE CACHE SIGNATURE
current_signature = create_signature(
    documents=documents,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    embedding_model=EMBEDDING_MODEL_NAME
)


cache_valid = is_cache_valid(
    metadata_path=METADATA_PATH,
    chunks_path=CHUNKS_PATH,
    embeddings_path=EMBEDDING_PATH,
    current_signature=current_signature
)



# LOAD EMBEDDING MODEL
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



# LOAD OR REBUILD CACHE
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

    embedding_model, embeddings = embd_chunks(chunks, model_name=EMBEDDING_MODEL_NAME)

    save_chunks(chunks, CHUNKS_PATH)

    save_embeddings(embeddings, EMBEDDING_PATH)

    save_cache_metadata(current_signature, METADATA_PATH)

    print("Cache rebuilt successfully.")


print("\nNumber of chunks:", len(chunks))

print("Embedding shape:", embeddings.shape)




print(
    "\n"
    + "=" * 30
    + " Loading Reranker "
    + "=" * 30
)

reranker = CrossEncoder(RERANKER_MODEL_NAME)

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


    candidates = retrieve(
        query=query,
        model=embedding_model,
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