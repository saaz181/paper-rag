# Paper RAG

A local retrieval-augmented generation pipeline for answering questions about scientific papers. Paper RAG extracts PDF text, retrieves and reranks passages, and generates answers with document and page citations. It includes a 301-question retrieval benchmark with reproducible baseline results and failure diagnostics.


## Features

- PDF ingestion with page text and document metadata.
- Paragraph-based chunks with overlapping windows for long paragraphs and noise filtering.
- Normalized sentence embeddings, dense retrieval, and cross-encoder reranking.
- BM25/dense hybrid retrieval with reciprocal rank fusion; dense-only experiments remain available.
- Local answer generation with source labels and an insufficient-context instruction.
- Cached chunks and embeddings, validated against corpus and configuration signatures.
- Page, strict evidence, exact evidence, and fuzzy evidence evaluation, with category breakdowns.
- JSON reports with complete returned chunk text, per-matcher flags, and stage-specific failures.

## Pipeline and specifications

```text
PDFs → page extraction → cleaning and chunking → embeddings
                                                    ↓
Question → dense top 20 + BM25 top 20 → RRF top 20 → reranked top 3
                                                    ↓
                                      context → generated answer + citations
```

| Component | Default |
| --- | --- |
| PDF extraction | `pypdf`; physical pages numbered from 1 on new ingestion |
| Embedding model | `intfloat/multilingual-e5-large-instruct` |
| Embedding dimension | 1024 |
| Similarity | Cosine Similarity |
| Chunk size | Up to 150 whitespace-separated words |
| Overlap | 25 words within sliding windows for oversized paragraphs |
| Dense retrieval | Top 20 chunks; no minimum score threshold |
| Fusion | BM25/dense RRF, k=60, branch depth 20, fused budget 20 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Final retrieval results | Top 3 chunks; `--rerank-top-k 5` enables the top-five experiment |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` |
| Generation | Greedy decoding; at most 220 new tokens by default |
| Generation device | Apple MPS when available, otherwise CPU |
| Fuzzy evidence threshold | 0.80; configurable from the evaluation CLI |

Ordinary paragraphs are packed together up to the word limit. The overlap setting applies when splitting oversized paragraphs, rather than between every pair of chunks. Filtering removes short passages, reference-like text, and some URL-heavy or numeric-heavy chunks.

The context builder preserves retrieved chunks in reranked order, including distinct passages from the same page. It removes only repeated text from the same document/page pair, so multiple top-three chunks from one page can reach the generator.

## Installation

The retrieval evaluator was run locally with Python **3.14.7**, NumPy **2.5.4**, PyTorch **2.14.1**, Sentence Transformers **6.2.0**, Transformers **5.20.2**, pypdf **6.21.2**, and python-dotenv **1.2.4**. `requirements.txt` pins runtime dependencies, including model downloads and the Transformers PyTorch/chat-template extras needed for generation. Pip resolves transitive dependencies; this is not a full environment lock. A clean installation on other platforms has not been verified.

```bash
git clone https://github.com/saaz181/paper-rag.git
cd paper-rag
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate` instead.

Optionally copy `.env.example` to `.env` and set a Hugging Face token:

```dotenv
HF_TOKEN=your_hugging_face_token
```

The first use downloads model files. Inference runs locally after loading. The `.env` file, PDFs, extracted corpus, model cache, and generated files under `runs/` stay out of Git. Benchmark labels and aggregate reference snapshots are versioned. Fresh clones must generate their own full reports; local `runs/` paths in examples describe experiment output locations.

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

The script loads or rebuilds the cache, runs the example questions in its `queries` list, and prints generated answers and sources. Use `--query` for your question. Shared reference settings live in `src/retrieval_config.py`; CLI options override them for individual experiments.

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

Reports score the returned reranked passages. Top-three runs report Hit@1,
Hit@3 and MRR@3. Top-five runs additionally report Hit@5 and MRR@5; their legacy
`mrr` field and failure counts use cutoff five. The `mrr_cutoff` and
`failure_cutoff` configuration fields state these cutoffs explicitly. Historical
saved baselines use three passages and the MiniLM cross-encoder.

| Metric | Match rule |
| --- | --- |
| Page | Expected document and designated page |
| Strict page evidence | Designated page plus the original normalized anchor substring rule |
| Exact document evidence | Normalized anchor with token boundaries, anywhere in the expected document |
| Fuzzy document evidence | Anchor similarity of at least 0.80, anywhere in the expected document |

The document-wide matchers normalize Unicode ligatures, PDF line-break hyphenation, punctuation, and whitespace while preserving complete numeric tokens. Decimal points, signs, scientific exponents, and percentage markers stay attached: `61` cannot match `61.9`, `-61`, or `61%`. Comma thousands separators are removed, and `61 %` normalizes to `61%`. Fuzzy matching compares word windows allowing one joined or split word. Pure numeric anchors require exact matches; other fuzzy windows must preserve the anchor's complete numeric tokens. Numeric spellings are compared lexically, rather than interpreting equivalent units or rounding values.

The current metric version is `document_fuzzy_numeric_v2`. Strict page evidence intentionally retains the original legacy normalization for historical comparison. Reports generated with older document evidence rules must be regenerated before comparing models; the comparison script rejects metric-version mismatches.

Fuzzy matching is a lexical heuristic, not a semantic evidence judge. A phrase or number recurring elsewhere in the document can produce a match without supporting the answer. Keep strict and exact scores alongside fuzzy scores when comparing experiments.

### Current retrieval reference

The shared defaults in `src/retrieval_config.py` select E5-large-instruct + BM25,
RRF k=60, branch depth 20, fused budget 20, and the MiniLM cross-encoder with
three final passages. On the 301-case development benchmark, fuzzy evidence
Hit@3 is **82.4% (248/301)** and MRR@3 is **0.704**. Returning five passages as a
separate experiment reaches **86.4% Hit@5 (260/301)**.

The [reference snapshot](benchmarks/baselines/e5_instruct_bm25_minilm_reference.json)
stores configuration, data hashes, category results, and the top-five experiment.
These are retrieval measurements, not answer correctness; reserve a separate
held-out set for final assessment after tuning on this development benchmark.

```bash
python evaluate.py --report runs/reference/evaluation_results.json
python evaluate.py --rerank-top-k 5 --report runs/reference_top5/evaluation_results.json
python compare_runs.py benchmarks/baselines/e5_instruct_bm25_minilm_reference.json runs/reference/evaluation_results.json
```

### Historical dense baseline

The table and snapshot below describe historical `document_fuzzy_v1` scores. They are retained as a record of that experiment and should not be compared directly with current numeric-aware scores. New numeric-aware model reports are saved under `runs/numeric_v2/`. The current [numeric-aware MiniLM baseline](benchmarks/baselines/minilm_150_25_numeric_v2.json) retains the same aggregate evidence scores, with failure counts of 9 chunking, 37 retrieval, and 32 reranking failures.

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
| `runs/hybrid/evaluation_results.json` | Default hybrid report: configuration, hashes, overall/category metrics, returned text, and match flags |
| `runs/hybrid/evaluation_results_failures.json` | Default hybrid failures, with diagnostics at the run's final cutoff |
| `benchmarks/baselines/minilm_150_25.json` | Fixed baseline summary, preserved across ordinary evaluation runs |

`--report` overrides the location; its failure file uses the same stem with
`_failures.json`. The legacy root-level report names apply when dense mode uses
its default path. Keep labels, metrics, chunking, and retrieval budgets fixed
when comparing experiments. If the metric changes, establish a new baseline.

### Compare embedding models

Run the same evaluation with a different embedding model and save each report
separately. Chunking (150 words, 25-word overlap), dense top 20, the cross-encoder,
reranked top 3, benchmark labels, and all scoring rules stay fixed.

```bash
python evaluate.py --embedding-model all-MiniLM-L6-v2 --retrieval-method dense --report runs/minilm/evaluation_results.json
python evaluate.py --embedding-model intfloat/e5-small-v2 --retrieval-method dense --report runs/e5/evaluation_results.json
python evaluate.py --embedding-model BAAI/bge-small-en-v1.5 --retrieval-method dense --report runs/bge/evaluation_results.json

python compare_runs.py runs/minilm/evaluation_results.json runs/e5/evaluation_results.json runs/bge/evaluation_results.json --output runs/comparison.txt
```

The **first report is the baseline**; every subsequent report is compared against
it. `compare_runs.py` uses only the Python standard library and never loads models.
It prints overall page/strict/exact/fuzzy metric changes, fuzzy failure-count
changes, per-category fuzzy Hit@1/Hit@3/MRR and failure deltas, and a descriptive
ranking. If both reports contain individual cases, it also shows questions gained
or lost under Evidence Hit@3. Deltas are absolute score differences, not relative
percent changes. A delta of `+0.051` means **5.1 percentage points**.

The comparator rejects mismatched chunks, benchmark hashes, rerankers, retrieval
depths, scoring settings, or category counts by default. `--allow-mismatch`
explicitly prints an exploratory comparison with warnings and suppresses the
ranking. It accepts the versioned summary-only baseline too:

```bash
python compare_runs.py benchmarks/baselines/minilm_150_25_numeric_v2.json runs/numeric_v2/e5/evaluation_results.json
```

E5-v2 and multilingual E5 small/base/large use `query: ` and `passage: ` prefixes,
as prescribed by [the E5 model card](https://huggingface.co/intfloat/e5-small-v2).
English BGE small/base/large variants use the query instruction recommended in
[the BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5), with no passage
prefix. Prefixes are applied only during encoding; the reranker receives the
original question and chunk text. Both prefixes are recorded in reports. These
are model-specific input formats within the embedding comparison.

Other Sentence Transformer models can be selected with `--embedding-model`.
Unknown models default to empty prefixes; use `--query-prefix` and
`--passage-prefix` according to their model cards. E5 instruct variants default
to a scientific-paper retrieval instruction in `Instruct: ...\nQuery: ` format,
with no passage prefix. Explicit prefix options override these defaults. A custom encoder
outside the Sentence Transformer interface may need additional integration.

Each model/format other than unprefixed MiniLM receives an isolated cache under
`cache/experiments/`, keyed by model, prefixes, corpus, and chunk settings.
Unprefixed MiniLM runs continue to use the legacy cache. `--cache-dir` can override
this location. Each custom report also gets a neighboring
`evaluation_results_failures.json`, keeping experiment failures separate.
Model-native dimensions and token limits may differ; chunking and retrieval
settings stay fixed. Changing a model downloads it on first use; running the
comparison script itself requires no downloads.

After models have been cached, evaluation can run without Hub access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate.py
```

### Try hybrid keyword and embedding retrieval

Hybrid mode searches the same chunks using both dense embeddings and BM25.
Each branch returns up to 20 candidates. Reciprocal rank fusion (RRF) combines
their rankings, keeps the best 20 unique chunks, and sends those to the existing
cross-encoder, which returns three passages. The generator uses those passages
as usual. Hybrid retrieval is the reference default.

BM25 helps passages with matching scientific terminology enter the candidate
pool; dense retrieval handles similarity in meaning. RRF adds `1 / (60 + rank)`
for each branch containing a chunk, using ranks starting at one. It avoids
directly mixing BM25 scores and cosine similarities. Ties favor dense order.
The local BM25 index uses k1=1.5, b=0.75, Unicode normalization, case folding,
word/decimal tokens, and PDF line-break dehyphenation; it does not stem or remove
stop words. It is built once per run without additional dependencies.

Start with MiniLM so that retrieval method is the only experimental change:

```bash
source venv/bin/activate
python evaluate.py --embedding-model all-MiniLM-L6-v2 --retrieval-method hybrid --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2 --rerank-top-k 3 --report runs/hybrid/minilm/evaluation_results.json
python compare_runs.py runs/numeric_v2/minilm/evaluation_results.json runs/hybrid/minilm/evaluation_results.json --output runs/hybrid/minilm/comparison.txt
```

The comparator displays each run's retrieval method and hybrid settings while
checking the shared chunk, benchmark, metric, reranker, and final candidate
budgets. Retrieval method, embedding model, and hybrid settings are experiment
variables: keep all but one fixed when attributing a change. Older reports
without a retrieval method are labeled dense. Reports include BM25 scores, RRF
scores, and branch ranks for returned hybrid results. `embedding_score` is null
when a passage only entered through BM25; RRF scores are not probabilities.

Use `--branch-top-k` and `--rrf-k` for later experiments. The final reranker input
budget stays at 20. Compare evidence Hit@3, MRR, and retrieval/reranking failures,
including questions gained and lost. Improvements are empirical; hybrid can
also displace useful dense results. Retrieval metrics do not establish answer
accuracy. Keep a held-out question set for final validation.

The first MiniLM hybrid run on the existing 301-case numeric-v2 benchmark
improved fuzzy evidence Hit@3 from 223/301 (74.1%) to 246/301 (81.7%), with
26 questions gained and 3 lost. Evidence MRR increased from 0.644 to 0.702.
Retrieval failures fell from 37 to 9; reranking failures rose from 32 to 37;
chunking failures stayed at 9. The comparator verified matching fixed settings
and data hashes against `runs/numeric_v2/minilm/evaluation_results.json`.
Local reports and the comparison are under `runs/hybrid/minilm/`.

Then ask a question through the full answer pipeline:

```bash
python src/rag.py --retrieval-method hybrid --rerank-top-k 3 --query "How does DPR use in-batch negatives?"
```

Existing cached embeddings can be reused. Add `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` before Python commands if all required models are already
cached. Generation additionally requires the Qwen generator model.

### Reranker and hybrid parameter experiments

The alternative `BAAI/bge-reranker-base` reranker uses the
[model card's CrossEncoder interface](https://huggingface.co/BAAI/bge-reranker-base).
The reference default is the MiniLM cross-encoder with three final passages.
Select BGE and top five explicitly for a reranker experiment.
Changing the reranker reuses the passage embeddings; it does not rebuild them.

Run one E5-instruct hybrid configuration with the BGE reranker:

```bash
python evaluate.py --embedding-model intfloat/multilingual-e5-large-instruct \
  --retrieval-method hybrid --reranker-model BAAI/bge-reranker-base \
  --rerank-top-k 5 --branch-top-k 20 --rrf-k 60 \
  --report runs/bge_top5/evaluation_results.json
```

The evaluator records per-question candidate identities and counts (dense,
BM25, overlap, unique union, and actual reranker input). It measures wall-clock
query encoding/dense search, BM25, fusion, retrieval, reranking, and total latency.
Overall and category reports include averages. Models receive one warm-up query;
loading, indexing, evidence-metric computation, and serialization are excluded.
These are single-pass descriptive timings; there are no cached reranker scores.
`--device cpu`, `--device mps`, or `--device cuda` fixes both models' device.

Run the complete controlled suite:

```bash
python run_reranker_experiments.py --output-dir runs/reranker_experiments_mps --device mps --resume
```

Use `--device cpu` and a separate output directory on CPU. All configurations use
E5-instruct embeddings and a fused candidate budget of 20. The suite compares:

- BGE and MiniLM rerankers at final cutoff five, with Hit@3 and Hit@5.
- BGE branch depths 10, 20, and 40, holding RRF=60 and final cutoff three.
- BGE RRF constants 1, 10, 60, and 100, holding branch depth=20 and final cutoff three.

The branch20/RRF60 top-three report is rescored from the prefix of the top-five
ranking, since both score the same 20 candidates. Its latency is shared with
that measured pass and explicitly labeled. Other suite configurations run
independently. Branch10 can produce fewer than 20 unique candidates due to
overlap; counts show this rather than assuming a full candidate budget.
`summary.md` and `summary.json` contain evidence scores, candidate counts,
latencies, and selected-candidate membership/order changes relative to
branch20/RRF60. Per-question changes include added/removed chunk identities.

Reranker files can live in a project-local Hugging Face cache; pass
`--reranker-cache-dir cache/models` (automatically detected for BGE when present).
The suite defaults to this location. Download the required model there once:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("BAAI/bge-reranker-base", cache_dir="cache/models", allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])'
```

Compare two same-cutoff reports while explicitly allowing a reranker experiment:

```bash
python compare_runs.py \
  runs/reranker_experiments_mps/minilm/branch20_rrf60_top5/evaluation_results.json \
  runs/reranker_experiments_mps/bge_base/branch20_rrf60_top5/evaluation_results.json \
  --vary reranker_model
```

The standard comparator still rejects undeclared reranker changes and differing
final/MRR cutoffs. The suite summary places Hit@3 and Hit@5 in separate columns.

A measured same-device comparison reached 86.4% Hit@5 with the MiniLM reranker
and 85.7% with BGE-base, so BGE remains an experiment. Recorded MPS latency means
were about 95 ms/query for MiniLM and 573 ms/query for BGE-base, including
retrieval and reranking after warm-up. These are single-pass host measurements.

## Tests

The project's **44 regression tests** require neither PDFs nor model downloads
(retrieval tests use the installed NumPy dependency):

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
│   ├── encoding.py           # Shared query formatting and cache selection
│   ├── lexical.py            # BM25 keyword retrieval
│   ├── retrieval_config.py   # Frozen reference defaults
│   ├── retriever.py          # Dense/hybrid retrieval, reranking, and context
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
├── compare_runs.py          # Controlled comparisons of saved model runs
├── run_reranker_experiments.py # Reranker, branch-depth, and RRF experiments
├── compare.sh               # Wrapper for one compatible run directory
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
