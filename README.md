# Paper RAG

A local retrieval-augmented generation pipeline for answering questions about scientific papers. Paper RAG extracts PDF text, retrieves and reranks passages, and generates answers with document and page citations. It includes a 301-question retrieval benchmark with reproducible baseline results and failure diagnostics.

## Features

- PDF ingestion with page text and document metadata.
- Paragraph-based chunks with overlapping windows for long paragraphs and noise filtering.
- Normalized sentence embeddings, dense retrieval, and cross-encoder reranking.
- Local answer generation with source labels and an insufficient-context instruction.
- Cached chunks and embeddings, validated against corpus and configuration signatures.
- Page, strict evidence, exact evidence, and fuzzy evidence evaluation, with category breakdowns.
- JSON reports with complete returned chunk text, per-matcher flags, and stage-specific failures.

## Pipeline and specifications

```text
PDFs → page extraction → cleaning and chunking → embeddings
                                                    ↓
Question → query embedding → dense top 20 → reranked top 3
                                                    ↓
                                      context → generated answer + citations
```

| Component | Default |
| --- | --- |
| PDF extraction | `pypdf`; physical pages numbered from 1 on new ingestion |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Similarity | Dot product of normalized embeddings, equivalent to cosine similarity |
| Chunk size | Up to 150 whitespace-separated words |
| Overlap | 25 words within sliding windows for oversized paragraphs |
| Dense retrieval | Top 20 chunks; no minimum score threshold |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Final retrieval results | Top 3 chunks |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` |
| Generation | Greedy decoding; at most 220 new tokens by default |
| Generation device | Apple MPS when available, otherwise CPU |
| Fuzzy evidence threshold | 0.80; configurable from the evaluation CLI |

Ordinary paragraphs are packed together up to the word limit. The overlap setting applies when splitting oversized paragraphs, rather than between every pair of chunks. Filtering removes short passages, reference-like text, and some URL-heavy or numeric-heavy chunks.

The context builder keeps the first returned chunk for each document/page pair. Consequently, multiple top-three chunks from one page do not all reach the generator.

## Installation

The retrieval evaluator was run locally with Python **3.14.7**, NumPy **2.5.4**, PyTorch **2.14.1**, Sentence Transformers **6.2.0**, Transformers **5.20.2**, pypdf **6.21.2**, and python-dotenv **1.2.4**. `requirements.txt` records the environment's pinned packages, including the direct runtime dependencies. A clean installation on other platforms has not been verified.

```bash
git clone https://github.com/saaz181/paper-rag.git
cd paper-rag
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate` instead.

Optionally copy `.env.example` to `.env` and set a Hugging Face token:

```dotenv
HF_TOKEN=your_hugging_face_token
```

The first use downloads model files. Inference runs locally after loading. The `.env` file, PDFs, extracted corpus, model-derived cache, and generated JSON reports stay out of Git. The benchmark labels and baseline summary are versioned.

## Run the pipeline

Run commands from the repository root.

### 1. Add PDF papers

```bash
mkdir -p data
```

Place text-based `.pdf` files directly in `data/`. Scanned papers need OCR first. Source PDFs are not distributed in this repository.

### 2. Extract the corpus

```bash
python -m src.ingest
```

This writes `data.json`, keyed by PDF filename, with page text and metadata. Repeat ingestion after adding or changing papers.

### 3. Generate answers

```bash
python src/rag.py
```

The script loads or rebuilds the cache, runs the example questions in its `queries` list, and prints generated answers and sources. Edit that list for your questions. Configure models, chunking, and retrieval depths using the constants near the top of `src/rag.py`; evaluation settings live in `evaluate.py`.

Source labels use the page numbers stored in `data.json`, for example `[2005.11401v4.pdf, p. 1]`. These are physical PDF positions, not printed journal page labels.

The cache signature covers extracted documents, chunk size, overlap, and embedding model name. Changing these inputs rebuilds the cache. If you change the chunking implementation itself, clear the local cache before running again; source code is not part of the signature.

## Retrieval benchmark

The bundled benchmark contains **301 curated cases across five papers and ten categories**. Cases include stable IDs, questions, reference answers, designated source pages, evidence anchors and spans, and original chunk-retention audits.

| Paper | Required PDF filename | Cases |
| --- | --- | ---: |
| Sentence-BERT | `1908.10084v1.pdf` | 50 |
| Dense Passage Retrieval | `2004.04906v3.pdf` | 60 |
| ColBERT | `2004.12832v2.pdf` | 60 |
| Retrieval-Augmented Generation | `2005.11401v4.pdf` | 60 |
| RAGChecker | `2408.08067v2.pdf` | 71 |

To reproduce this benchmark, obtain the exact PDF versions named above and place them in `data/`. Expected source SHA-256 hashes and page counts are recorded in [the manifest](benchmarks/manifest.json). The bundled cases refer to these papers; evaluating a different corpus requires corresponding labels.

```bash
python -m src.ingest
python -m benchmarks.validate
python evaluate.py
```

Validation checks PDF hashes, exact evidence spans, unique questions, category counts, and page-number conversions. It verifies provenance and structure, rather than independently establishing semantic label correctness.

The saved baseline used a legacy **0-based** extracted corpus with **300 retained chunks**. New ingestion uses **1-based** pages. The evaluator detects either convention and adjusts case labels. Exact baseline reproduction also requires matching corpus and chunk hashes; page-base conversion alone does not guarantee identical extraction or chunking.

### Metrics

All Hit@k and MRR values are calculated over the reranked top three chunks; MRR is therefore truncated at rank 3.

| Metric | Match rule |
| --- | --- |
| Page | Expected document and designated page |
| Strict page evidence | Designated page plus the original normalized anchor substring rule |
| Exact document evidence | Normalized anchor with token boundaries, anywhere in the expected document |
| Fuzzy document evidence | Anchor similarity of at least 0.80, anywhere in the expected document |

The document-wide matchers normalize Unicode ligatures, PDF line-break hyphenation, punctuation, and whitespace. Fuzzy matching compares word windows allowing one joined or split word. Pure numeric anchors require exact matches; other fuzzy windows must preserve the anchor's numeric tokens.

Fuzzy matching is a lexical heuristic, not a semantic evidence judge. A phrase or number recurring elsewhere in the document can produce a match without supporting the answer. Keep strict and exact scores alongside fuzzy scores when comparing experiments.

### Saved baseline

Configuration: MiniLM embeddings, MiniLM cross-encoder, 150-word chunks, 25-word overlap, dense top 20, reranked top 3, and fuzzy threshold 0.80.

| Metric | Hit@1 | Hit@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| Page | 0.575 | 0.774 | 0.664 |
| Strict page evidence | 0.482 | 0.664 | 0.563 |
| Exact document evidence | 0.502 | 0.668 | 0.577 |
| Fuzzy document evidence | 0.561 | 0.741 | 0.644 |

Under fuzzy document matching, **223/301** cases have evidence in the top three. The remaining **78** cases split into **8 chunking**, **37 dense retrieval**, and **33 reranking** failures. Each failure is assigned to the first stage that lacks a matching anchor, using current chunks rather than trusting the original audit.

These metrics describe retrieval, not generated-answer accuracy. Higher scores under a looser matcher reflect a change in scoring, rather than an improvement to the retrieval models.

The versioned [baseline snapshot](benchmarks/baselines/minilm_150_25.json) includes corpus, chunk, and benchmark hashes, metric settings, category results, and failure counts. See [the benchmark guide](benchmarks/README.md) for detailed coverage and limitations.

### Reports and evaluation options

```bash
python evaluate.py --fuzzy-threshold 0.85 --report evaluation_results_085.json
```

| Output | Contents |
| --- | --- |
| `evaluation_results.json` | Configuration, hashes, overall/category metrics, all cases, full returned text, and match flags |
| `evaluation_failures.json` | Cases that fail fuzzy document evidence matching, with stage diagnostics |
| `benchmarks/baselines/minilm_150_25.json` | Fixed baseline summary, preserved across ordinary evaluation runs |

Use identical benchmark labels, metric settings, chunking, and retrieval depths when comparing future embedding or reranking changes. If the metric changes, establish a new baseline.

After models have been cached, evaluation can run without Hub access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate.py
```

## Tests

The evaluator's **11 regression tests** run with the Python standard library and require neither PDFs nor model downloads:

```bash
python3 -m unittest discover -s tests -v
```

They cover benchmark metadata, page-base conversion, label validation, PDF normalization, cross-page matching, fuzzy thresholds, numeric safeguards, stage diagnostics, and report aggregation. PDF provenance validation separately requires the source files and `pypdf`:

```bash
python -m benchmarks.validate
```

## Repository structure

```text
paper-rag/
├── src/
│   ├── ingest.py             # PDF extraction
│   ├── chunking.py           # Cleaning, chunking, and noise filtering
│   ├── embeddings.py         # Embedding generation and persistence
│   ├── retriever.py          # Dense retrieval, reranking, and context
│   ├── generator.py          # Local language model and citation prompt
│   ├── cache.py              # Cache signatures and persistence
│   └── rag.py                # End-to-end example runner
├── benchmarks/
│   ├── cases.json            # Full benchmark labels and source evidence
│   ├── manifest.json         # Source hashes and dataset metadata
│   ├── test_cases.py         # Full and minimal case loaders
│   ├── metrics.py            # Page and evidence matching
│   ├── validate.py           # Offline source-provenance validation
│   └── baselines/            # Versioned baseline summary
├── tests/test_evaluation.py  # Model-free regression tests
├── evaluate.py              # Benchmark runner and JSON reporting
├── app.py                   # Placeholder for a future interface
├── requirements.txt
├── .env.example
├── data/                    # Local PDFs; ignored
├── data.json                # Extracted corpus; ignored
└── cache/                   # Chunks and embeddings; ignored
```

## Scope and limitations

The local pipeline and retrieval evaluation are implemented. `app.py` remains an empty placeholder; the repository currently runs through scripts rather than a web interface.

- PDF extraction artifacts, tables, equations, and noise filtering can remove useful evidence. OCR is not included.
- Benchmark labels are curated drafts with one designated page, not exhaustive relevance judgments. Alternative valid evidence may be missed.
- The benchmark does not measure generated-answer correctness, citation faithfulness, or completeness. Check generated claims against the source papers.
- Cross-document synthesis, unanswerable questions, multilingual queries, and image-only evidence are outside the bundled benchmark's scope.
- CUDA selection is not implemented for the generator; it explicitly selects MPS or CPU. Resource use depends on model files, device, and corpus size; no memory or latency guarantee is provided.
- Repeated tuning on the same 301 cases can overfit this development benchmark. Use a separate held-out set for final quality claims.

Possible next experiments include hybrid lexical/dense retrieval and alternative embedding or reranking models, compared against the fixed baseline with unchanged scoring rules.
