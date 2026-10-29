import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.test_cases import load_benchmark
from compare_runs import compatibility_issues, load_report
from evaluate import analyze_case, detect_page_base, summarize

EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
BGE = "BAAI/bge-reranker-base"
MINILM = "cross-encoder/ms-marco-MiniLM-L6-v2"


def identity(result):
    return result["document"], result["page"], result["chunk_id"]


def candidate_changes(baseline, candidate):
    left = {row["id"]: row for row in baseline["cases"]}
    right = {row["id"]: row for row in candidate["cases"]}

    if left.keys() != right.keys():
        raise ValueError("Cannot compare candidate selections for different question sets")

    changes = []
    for case_id, before in left.items():
        after = right[case_id]
        before_order = list(map(identity, before["candidates"]))
        after_order = list(map(identity, after["candidates"]))
        a, b = set(before_order), set(after_order)

        changes.append({"id": case_id, "membership_changed": a != b,
                        "order_changed": before_order != after_order,
                        "added": sorted(b - a), "removed": sorted(a - b),
                        "jaccard": len(a & b) / len(a | b) if a | b else 1.0})

    return {"questions": len(changes),
            "membership_changed": sum(row["membership_changed"] for row in changes),
            "order_changed": sum(row["order_changed"] for row in changes),
            "average_jaccard": sum(row["jaccard"] for row in changes) / len(changes),
            "cases": changes}


def project_top_three(report, cache_dir):
    config = {**report["config"], "rerank_top_k": 3, "mrr_cutoff": 3, "failure_cutoff": 3,
              "projection": "top3_prefix_of_top5_same_20_candidate_scoring_pass",
              "latency_note": "Shared measured top5 pass; no independent top3 timing run"}

    chunks = json.loads((cache_dir / "chunks.json").read_text())
    lookup = {identity(chunk): chunk for chunk in chunks}
    grouped = defaultdict(list)

    for chunk in chunks:
        grouped[chunk["document"]].append(chunk)

    cases = {case["id"]: case for case in load_benchmark(page_base=config["page_base"])}
    rows = []
    for source in report["cases"]:
        case = cases[source["id"]]
        candidates = [lookup[identity(candidate)] for candidate in source["candidates"]]
        row = analyze_case(case, grouped[case["expected_document"]], candidates,
                           source["retrieved_results"][:3], config["fuzzy_threshold"], 3)
        row.update({key: source[key] for key in ("candidate_counts", "latency_ms", "candidates")})
        rows.append(row)
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    return {"config": config, "overall": summarize(rows),
            "categories": {key: summarize(group) for key, group in sorted(categories.items())},
            "cases": rows}


def save_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    path.with_name(path.stem + "_failures.json").write_text(json.dumps(
        [row for row in report["cases"] if row["failure_type"]], indent=2, ensure_ascii=False) + "\n")


def write_summary(reports, output_dir):
    base = reports["bge_base/branch20_rrf60_top3"]
    results, changes = [], {}
    for name, report in reports.items():
        issues = compatibility_issues(base, report,
                                      vary=("reranker_model", "rerank_top_k", "mrr_cutoff"))
        if issues:
            raise ValueError(f"{name}: incompatible benchmark: {issues}")
        overall = report["overall"]
        results.append({"name": name, "config": report["config"],
                        "evidence": overall["evidence"], "exact_evidence": overall["exact_evidence"],
                        "failure_counts": overall["failure_counts"]["evidence"],
                        "candidate_counts": overall["average_candidate_counts"],
                        "latency_ms": overall["average_latency_ms"]})
        if report["config"]["rerank_top_k"] == 3:
            changes[name] = candidate_changes(base, report)
    (output_dir / "summary.json").write_text(json.dumps(
        {"runs": results, "candidate_changes_vs_branch20_rrf60": changes}, indent=2) + "\n")
    lines = ["# Reranker and hybrid retrieval experiments", "",
             "Embedding model: intfloat/multilingual-e5-large-instruct. Fused candidate budget: 20.",
             "All runs use the same 301 questions, chunks, and evidence matching rules.", "",
             "| Run | Hit@1 | Hit@3 | Hit@5 | MRR@3 | MRR@5 | Mean union | Mean reranker input | Retrieval ms | Rerank ms | Total ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        evidence, counts, timing = row["evidence"], row["candidate_counts"], row["latency_ms"]
        hit5 = f"{evidence['hit_at_5']:.3f}" if "hit_at_5" in evidence else "—"
        mrr5 = f"{evidence['mrr_at_5']:.3f}" if "mrr_at_5" in evidence else "—"
        lines.append(f"| {row['name']} | {evidence['hit_at_1']:.3f} | {evidence['hit_at_3']:.3f} | "
                     f"{hit5} | {evidence['mrr_at_3']:.3f} | {mrr5} | {counts['union']:.2f} | "
                     f"{counts['reranker_input']:.2f} | {timing['retrieval']:.2f} | "
                     f"{timing['rerank']:.2f} | {timing['total']:.2f} |")
    lines += ["", "## Candidate selection versus branch=20, RRF=60", "",
              "Membership compares the fused candidates before the reranker; order includes membership changes.", "",
              "| Run | Changed sets / 301 | Changed ordered lists / 301 | Mean Jaccard |",
              "|---|---:|---:|---:|"]
    for name, change in changes.items():
        lines.append(f"| {name} | {change['membership_changed']} | {change['order_changed']} | "
                     f"{change['average_jaccard']:.4f} |")

    lines += ["", "Latency is wall-clock time for sequential queries after one warm-up query, with no score cache.",
              "It excludes model loading, index construction, scoring metrics, and report serialization; includes query encoding.",
              "One pass per configuration: descriptive timings on this host, not repeated performance estimates.",
              "The branch20/RRF60 top3 row is a prefix of the top5 scoring pass and shares its measured latency.",
              "MRR cutoffs are shown separately; failure counts in each JSON refer to its final cutoff.",
              "Branch10 can supply fewer than 20 unique candidates because of overlap between branches.",
              "Higher retrieval scores on this development benchmark do not establish answer accuracy or held-out gains.", ""]
    (output_dir / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reranker_experiments"))
    parser.add_argument("--reranker-cache-dir", type=Path, default=Path("cache/models"))
    parser.add_argument("--resume", action="store_true", help="Reuse validated completed reports")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir = args.output_dir.resolve()
    args.reranker_cache_dir = args.reranker_cache_dir.resolve()
    documents = json.loads((root / "data.json").read_text())
    page_base = detect_page_base(documents)
    docs_hash = hashlib.sha256(json.dumps(documents, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    benchmark_hash = hashlib.sha256((root / "benchmarks/cases.json").read_bytes()).hexdigest()
    plan = [("bge_base/branch20_rrf60_top5", BGE, 20, 60, 5),
            ("minilm/branch20_rrf60_top5", MINILM, 20, 60, 5),
            ("bge_base/branch10_rrf60_top3", BGE, 10, 60, 3),
            ("bge_base/branch40_rrf60_top3", BGE, 40, 60, 3),
            *[(f"bge_base/branch20_rrf{k}_top3", BGE, 20, k, 3) for k in (1, 10, 100)]]
    reports = {}
    for name, reranker, branch, rrf, cutoff in plan:
        path = args.output_dir / name / "evaluation_results.json"
        if args.resume and path.exists():
            report = load_report(path)
            config = report["config"]
            expected = {"embedding_model": EMBEDDING_MODEL, "reranker_model": reranker,
                        "rerank_top_k": cutoff, "retrieval_top_k": 20, "page_base": page_base,
                        "documents_sha256": docs_hash, "benchmark_sha256": benchmark_hash,
                        "retrieval_method": "hybrid", "fuzzy_threshold": .8,
                        "metric_version": "document_fuzzy_numeric_v2"}

            if any(config.get(key) != value for key, value in expected.items()) or any(
                    config["hybrid_config"][key] != value for key, value in
                    (("branch_top_k", branch), ("rrf_k", rrf))):

                raise ValueError(f"{path}: existing report does not match requested experiment")

            if args.device and any(config.get(key, "").split(":")[0] != args.device
                                   for key in ("embedding_device", "reranker_device")):
                raise ValueError(f"{path}: existing report used a different device")
            print(f"Reusing {name}", flush=True)

        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(root / "evaluate.py"), "--embedding-model", EMBEDDING_MODEL,
                       "--retrieval-method", "hybrid", "--reranker-model", reranker,
                       "--branch-top-k", str(branch), "--rrf-k", str(rrf),
                       "--rerank-top-k", str(cutoff), "--report", str(path)]
            if reranker == BGE:
                command += ["--reranker-cache-dir", str(args.reranker_cache_dir)]
            if args.device:
                command += ["--device", args.device]
            print(f"Running {name}; log: {path.parent / 'evaluation.log'}", flush=True)
            with (path.parent / "evaluation.log").open("w") as log:
                subprocess.run(command, cwd=root, env=os.environ.copy(), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            report = load_report(path)
        reports[name] = report
        if name == "bge_base/branch20_rrf60_top5":
            cache_dir = root / "cache" / "experiments" / report["config"]["corpus_signature"]
            projected = project_top_three(report, cache_dir)
            projection_name = "bge_base/branch20_rrf60_top3"
            save_report(projected, args.output_dir / projection_name / "evaluation_results.json")
            reports[projection_name] = projected
    write_summary(reports, args.output_dir)
    print(f"Summary saved to {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
