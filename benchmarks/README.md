# Five-paper RAG benchmark

301 curated question-answer cases grounded in the extracted text of the five PDFs in `data/`. Each case asks for information available on one designated source page. The set covers main-text methods, experimental settings, results, limitations, and substantive appendices. References and acknowledgments are not used as question material. Numeric table entries are included where the extracted text preserves the answer; no answer is inferred solely from a plot or image.

## Coverage

| Paper | Local PDF | Cases | Main coverage |
| --- | --- | ---: | --- |
| Sentence-BERT | `1908.10084v1.pdf` | 50 | Sentence representations, pooling, objectives, NLI training, STS, argument similarity, transfer tasks, and efficiency |
| Dense Passage Retrieval | `2004.04906v3.pdf` | 60 | Dual encoders, negative sampling, Wikipedia preprocessing, hybrid retrieval, ablations, reader design, and runtime |
| ColBERT | `2004.12832v2.pdf` | 60 | Late interaction, MaxSim, query augmentation, offline indexing, approximate retrieval, and storage-quality trade-offs |
| Retrieval-Augmented Generation | `2005.11401v4.pdf` | 60 | Parametric/non-parametric memory, sequence/token formulations, decoding, QA, verification, hot-swapping, and collapse |
| RAGChecker | `2408.08067v2.pdf` | 71 | Claim-level metrics, retrieval/generation diagnosis, benchmark construction, human evaluation, tuning, and limitations |

### What each paper contributes to this benchmark

**Sentence-BERT:** Tests whether retrieval can distinguish meaningful sentence embeddings from unadapted BERT outputs and recover the details behind classification, regression, and triplet training. The paper also supplies useful counterexamples: performance on a supervised transfer task does not automatically imply that cosine similarity between embeddings works well, and strong within-topic argument similarity does not guarantee cross-topic generalization.

**DPR:** Supplies questions about how dense retrieval is actually trained, especially the role of in-batch and BM25 hard negatives. Its experiments are useful for testing nuanced answers: dense retrieval can overcome vocabulary mismatch but still struggle with rare salient phrases, and SQuAD's lexical overlap favors BM25. Retrieval, indexing, and reader costs must be kept distinct.

**ColBERT:** Tests fine-grained architectural understanding. A correct answer should distinguish token-level late interaction from a single-vector dual encoder and from joint query-document cross-attention. The indexing and reranking sections expose common confusions between offline work, online query work, candidate filtering, and exact refinement.

**Original RAG:** Supplies questions about latent-document marginalization, the difference between per-sequence and per-token document use, fixed document encoders, and decoding approximations. Its hot-swapping and collapse experiments test whether answers retain the conditions and limitations of the reported findings rather than treating retrieval as a universal improvement.

**RAGChecker:** Extends coverage beyond whether a relevant page was found. Its metrics distinguish retrieved coverage, generated coverage, faithfulness, copying irrelevant information, and unsupported generation. Its analysis emphasizes that stronger retrieval can improve factual coverage while also exposing a generator to more noise, and that faithfulness alone does not establish answer correctness.

## Files

- `test_cases.py`: loader exposing `test_cases` in the exact three-field shape used by `evaluate.py`.
- `cases.json`: full cases with stable IDs, categories, reference answers, source evidence, and chunk-retention diagnostics.
- `manifest.json`: counts, source titles, PDF hashes, page conventions, and audit settings.
- `validate.py`: checks source hashes, page labels, evidence offsets, uniqueness, and loader conversion against the local PDFs.

## Use with the evaluator

Run `python evaluate.py` from the project root. It loads the full cases with
`load_benchmark()`, preserving categories, IDs, evidence, and audit metadata.
It infers the page base from `data.json` and rejects missing evidence labels.

`load_test_cases()` and `test_cases` remain available for consumers needing
only `question`, `expected_document`, and `expected_page`. These minimal cases
cannot support evidence metrics or category breakdowns.

The evaluator reports four sets of Hit@1, Hit@3, and MRR over the reranked
top 3 chunks:

- **Page:** exact designated document and page, unchanged.
- **Strict page evidence:** the previous designated-page metric with its original
  normalization, preserved for direct comparison with earlier runs.
- **Exact document evidence:** normalized anchor match anywhere in the expected
  document. Unicode ligatures, PDF line-break hyphenation, punctuation and spaces
  are normalized, with token boundaries preserved.
- **Evidence (fuzzy document):** the same document constraint, with `SequenceMatcher`
  similarity at least 0.80 over windows of the anchor's word count, plus or minus
  one word to accommodate joined/split PDF words. Pure numeric anchors require
  exact matches; other fuzzy windows must preserve the anchor's numeric tokens.

Fuzzy scores measure lexical similarity, not semantic support or correctness.
A paraphrase can still fail, and a similar phrase can still be irrelevant. In
particular, removing the page constraint allows recurring short phrases or numbers
elsewhere in the same document to match. Keep the strict and exact results beside
the fuzzy score and review disputed labels. An increase caused by changing the
metric is not a retrieval improvement.

Failure analysis uses the same matcher as its corresponding score, checking all
current chunks in the expected document before dense candidates and reranked
results. Stored designated-page chunk audits remain provenance metadata.
`evaluation_failures.json` contains failures under the fuzzy document matcher;
`evaluation_results.json` contains configuration, corpus/chunk/benchmark hashes,
overall and per-category scores, failure counts for each evidence matcher, and
all cases with complete returned chunk text and per-matcher flags. MRR is cut off
at the reranker depth (3).

```bash
python evaluate.py
python evaluate.py --fuzzy-threshold 0.85 --report evaluation_results_085.json
python3 -m unittest discover -s tests -v
```

The fixed MiniLM baseline is saved in `baselines/minilm_150_25.json`. Subsequent
runs write the latest report without overwriting this snapshot. Compare future
retrieval/reranking experiments using the same benchmark, matching settings,
chunking, and retrieval depths. Recompute baselines whenever the metric changes.

### Page-number convention

`expected_page` is **0-based by default**, matching the current local `data.json` and cached chunks. `pdf_page` in the full JSON is always the **1-based physical page position** in the PDF, including title pages. Neither field refers to an independently printed journal page number.

The current ingestion source starts numbering at 1. If you regenerate `data.json` with that source, use:

```python
from benchmarks.test_cases import load_test_cases

test_cases = load_test_cases(page_base=1)
```

Inspect an actual chunk's `page` value before choosing the convention. Rebuild stale caches after regenerating source data. Mixing conventions will produce false misses even when the correct page is retrieved.

## Reference answers and evidence

```python
from benchmarks.test_cases import load_benchmark

cases = load_benchmark(page_base=0)
case = cases[0]
print(case["id"], case["question"])
print(case["reference_answer"])
print(case["expected_document"], case["pdf_page"])
print(case["evidence"]["quote"])
```

`evidence.quote` is an exact window from unmodified `pypdf` page extraction around a selected evidence anchor. `start_char` and `end_char` are Python string offsets into that page text, with an exclusive end. Windows include surrounding context and may start or finish mid-sentence. They preserve extraction artifacts such as ligatures, hyphenated line breaks, and ColBERT's unusual encoded character sequences.

Reference answers are concise semantic targets, not mandatory exact-match strings. All labels were drafted from the local paper text. Automated checks establish source provenance and structure; they do not independently certify that each reference answer is complete or correct. Review disputed cases against the evidence before treating them as final gold labels.

## Categories

| Category | Cases |
| --- | ---: |
| Method | 77 |
| Comparison | 41 |
| Numerical | 38 |
| Concept | 27 |
| Training | 25 |
| Metric | 24 |
| Dataset | 21 |
| Result | 19 |
| Ablation | 17 |
| Limitation | 12 |

These are topic labels, not calibrated difficulty levels. Some method and training questions also require numeric details. Many questions name their target model or paper, making source selection easier than an unanchored user query. This is a useful development benchmark, not a comprehensive test of open-ended research QA.

## Chunk-retention audit

The audit uses the existing `chunk_docs` function with `max_words=150` and `overlap=25`, matching the pipeline defaults. Every designated page produces at least one retained chunk. Selected evidence anchors occur within individual retained chunks for **291 of 301 cases**.

The remaining ten cases are:

```text
sbert_009       sbert_048       dpr_022         colbert_049
colbert_052     rag_041         rag_055         ragchecker_039
ragchecker_041  ragchecker_042
```

A missing anchor does **not** establish that the answer is unavailable: the anchor may cross a chunk boundary, occur in a removed table, or have another supporting passage. Conversely, retaining an anchor does not prove that every detail needed for the answer survived. These flags help investigate preprocessing problems; they are not retrieval scores or semantic answerability labels. Recompute this diagnostic if chunking or source data changes.

## Scoring recommendations

1. Report document Hit@k and designated-page Hit@k separately, before and after reranking. A document match is weaker than retrieving its answer-bearing page.
2. Treat designated-page labels as one known supporting page, **not an exhaustive relevance judgment**. A valid answer may also appear on another page or in another paper; review such cases instead of automatically interpreting a strict miss as irrelevant retrieval.
3. For generator evaluation, compare generated claims against `reference_answer` and the retrieved evidence. Score completeness and unsupported claims separately. Exact text equality is inappropriate for most reference answers.
4. Track whether answer evidence survives preprocessing before attributing a failure to embeddings, reranking, or generation.
5. Break results down by paper and category. Use macro-averaging across papers if you want equal paper weighting: the overall average otherwise gives RAGChecker more weight than Sentence-BERT.
6. Do not ingest benchmark questions, answers, or evidence windows as retrieval documents. If tuning repeatedly against these cases, reserve a separate held-out set; this set is not an independent final test after that tuning.

Cross-document synthesis, unanswerable questions, adversarial false premises, multilingual queries, and image-only evidence are outside this set's scope. They need different labels or scoring and should be added as separate suites rather than forced into the single-document/single-page format.

## Validate locally

From the repository root, using an environment with `pypdf` installed:

```bash
python -m benchmarks.validate
```

Validation loads no embedding or generation models, makes no network requests, and does not run the RAG pipeline. It verifies all 301 evidence spans against the PDFs and checks both page-number conventions. No retrieval or answer-quality score has been claimed for this dataset.

## Comparing embedding runs

Use `evaluate.py --embedding-model MODEL --report runs/MODEL/evaluation_results.json`
with the same corpus and scoring settings, then run:

```bash
python compare_runs.py runs/minilm/evaluation_results.json runs/e5/evaluation_results.json runs/bge/evaluation_results.json
```

The first file is the baseline. Configuration and input fingerprints must match
apart from model-specific embedding settings. Overall metrics, category evidence
scores, failure changes, and (with full reports) paired case gains/losses are
printed. The model-dependent `corpus_signature` is not used as a fixed input
fingerprint; chunk and benchmark hashes are checked instead. See the root README
for model prefixes, cache isolation, and CLI options.

## Numeric-aware evidence scoring

Current exact and fuzzy document evidence use metric version
`document_fuzzy_numeric_v2`. Normalization preserves signs, decimal values,
scientific exponents, and `%` markers as part of complete numeric tokens. For
example, `61` does not match `61.9`, `-61`, or `61%`; `61 %` matches `61%`.
Comma thousands separators are normalized (`1,000` matches `1000`). Fuzzy
windows must preserve these complete numeric tokens. This is lexical matching,
not numeric reasoning about unit conversions or rounding.

Strict page evidence retains its legacy rules. Old baseline snapshots remain
historical records; compare current models using newly generated reports with
the same metric version. The numeric-aware reruns and their comparison live in
`runs/numeric_v2/` locally.

The current MiniLM baseline is `baselines/minilm_150_25_numeric_v2.json`. Its
fuzzy evidence scores remain Hit@1 0.561, Hit@3 0.741, and MRR 0.644, but the
numeric correction changes its stage diagnostics to 9 chunking, 37 retrieval,
and 32 reranking failures. The old `baselines/minilm_150_25.json` is a historical
snapshot under v1 rules.
