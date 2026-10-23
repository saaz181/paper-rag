# Paper RAG

A work-in-progress retrieval-augmented generation (RAG) project for answering questions about scientific papers. It extracts text from local PDFs, retrieves relevant passages, reranks them, and generates answers with document and page citations.

## Current pipeline

1. **Ingest PDFs:** extract page text and document metadata with `pypdf`.
2. **Prepare passages:** clean text, create paragraph-based chunks, and filter noisy passages.
3. **Embed and retrieve:** encode passages with `all-MiniLM-L6-v2` and rank them by similarity to the question.
4. **Rerank:** score the top 20 candidates with `cross-encoder/ms-marco-MiniLM-L6-v2` and keep the top 3.
5. **Generate:** use `Qwen/Qwen2.5-1.5B-Instruct` to answer from the retrieved context, with source labels such as `[paper.pdf, p. 1]`.

Chunks and embeddings are cached locally. A signature based on the extracted documents, chunk settings, and embedding model determines when the cache is rebuilt. Generation uses Apple MPS when available and otherwise runs on the CPU.

## Setup

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/saaz181/paper-rag.git
cd paper-rag
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pypdf python-dotenv sentence-transformers transformers
```

The additional install command includes packages imported by the code that are not yet listed in `requirements.txt`. Dependency setup is still being refined; the pinned requirements have not been verified in a fresh environment.

If needed, create a local `.env` file for a Hugging Face token:

```dotenv
HF_TOKEN=your_hugging_face_token
```

The first model load requires internet access to download model files. Models run locally after loading. The `.env` file is ignored by Git.

## Usage

Run all commands from the repository root.

### 1. Add papers

```bash
mkdir -p data
```

Place your `.pdf` files directly in `data/`. Ingestion extracts existing PDF text; scanned image-only papers need OCR before use. Papers are not included in this repository.

### 2. Extract document text

```bash
python -m src.ingest
```

This writes `data.json` with extracted text and metadata. Rerun ingestion whenever you add or change papers.

### 3. Run the RAG pipeline

```bash
python src/rag.py
```

The script loads or rebuilds the cache, runs the example questions defined in its `queries` list, and prints answers and sources. Edit that list to ask your own questions. Model names, chunk size, overlap, and retrieval settings are defined near the top of `src/rag.py`.

The default questions explore retrieval-augmented generation, so use papers relevant to that topic or adapt the questions to your documents.

## Retrieval evaluation

```bash
python evaluate.py
```

The evaluation script reports retrieval Hit@3 for a small set of manually defined questions and expected document/page pairs. Update `test_cases` to match your papers before using the result.

**Current limitation:** the bundled evaluation cases use page `0`, while ingestion numbers pages starting at `1`. Correct the expected pages for your corpus before interpreting the scores. This evaluates retrieval, not generated-answer quality.

## Project structure

```text
paper-rag/
├── src/
│   ├── ingest.py       # PDF text and metadata extraction
│   ├── chunking.py     # Text cleaning and passage creation
│   ├── embeddings.py   # Passage embeddings and NumPy persistence
│   ├── retriever.py    # Similarity search, reranking, and context assembly
│   ├── generator.py    # Local language model and answer prompts
│   ├── cache.py        # Cache signatures and persistence helpers
│   └── rag.py          # End-to-end example pipeline
├── evaluate.py         # Retrieval evaluation script
├── app.py              # Placeholder for a future application interface
├── requirements.txt
├── data/               # Local PDFs (ignored)
├── data.json           # Extracted documents (ignored)
└── cache/              # Generated chunks and embeddings (ignored)
```

## Development status

This repository is an early development checkpoint. The implementation includes ingestion, chunking, embedding retrieval, reranking, caching, and prompted answer generation. The application interface is still a placeholder, dependency setup needs validation, and evaluation coverage is limited. Generated answers and citations should be checked against the original papers.
