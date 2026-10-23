from src.retriever import retrieve, rerank
import os
from src.chunking import chunk_docs
from src.embeddings import embd_chunks
import json
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_JSON = "data.json"

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

with open(DATA_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)


chunks = chunk_docs(documents,
        max_words=CHUNK_SIZE,
        overlap=OVERLAP
    )

embedding_model, embeddings = embd_chunks(chunks, model_name=EMBEDDING_MODEL_NAME)

reranker = CrossEncoder(RERANKER_MODEL_NAME)


test_cases = [
    {
        "question": "What is retrieval-augmented generation?",
        "expected_document": "2005.11401v4.pdf",
        "expected_page": 0
    },
    {
        "question": (
            "What is the difference between parametric "
            "and non-parametric memory in RAG?"
        ),
        "expected_document": "2005.11401v4.pdf",
        "expected_page": 0
    }
]



def retrieval_hit(results, expected_document, expected_page):
    for result in results:
        if (
            result["document"] == expected_document
            and result["page"] == expected_page
        ):
            return True

    return False



correct = 0

for case in test_cases:

    query = case["question"]

    candidates = retrieve(
        query=query,
        model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=20,
        min_score=None
    )

    results = rerank(
        query=query,
        candidates=candidates,
        reranker=reranker,
        top_k=3
    )

    hit = retrieval_hit(
        results,
        expected_document=case["expected_document"],
        expected_page=case["expected_page"]
    )

    if hit:
        correct += 1

    print(query)
    print("Retrieval hit:", hit)
    print()




accuracy = correct / len(test_cases)

print(
    "Retrieval Hit@3:",
    accuracy
)