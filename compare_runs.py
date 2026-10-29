import argparse
import json
import math
from pathlib import Path

METRICS = {
    "evidence": "Evidence", "exact_evidence": "Exact evidence",
    "strict_evidence": "Strict evidence", "page": "Page",
}
SCORES = {"hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "mrr": "MRR"}
FAILURES = ("chunking_failure", "retrieval_failure", "reranking_failure")
FIXED_CONFIG = (
    "reranker_model", "chunk_size_words", "overlap_words", "retrieval_top_k",
    "rerank_top_k", "page_base", "chunks_sha256", "benchmark_sha256",
    "metric_version", "fuzzy_threshold", "numeric_fuzzy_guard",
    "fuzzy_window_word_tolerance", "evidence_scope", "normalization", "mrr_cutoff",
)


def load_report(path):
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report["config"]["embedding_model"], str):
            raise ValueError("embedding_model must be a string")
        groups = [report["overall"], *report["categories"].values()]
        cutoff = report["config"].get("failure_cutoff", report["config"].get("rerank_top_k", 3))
        if cutoff not in (3, 5):
            raise ValueError("failure cutoff must be 3 or 5")
        for group in groups:
            total = group["total"]
            if type(total) is not int or total <= 0:
                raise ValueError("question counts must be positive integers")
            for metric in METRICS:
                for score in (*SCORES, *(["hit_at_5"] if cutoff == 5 else [])):
                    value = group[metric][score]
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"{metric}.{score} must be a finite number in [0, 1]")
            for metric in ("evidence", "exact_evidence", "strict_evidence"):
                counts = group["failure_counts"][metric]
                if any(type(counts[key]) is not int or not 0 <= counts[key] <= total for key in FAILURES):
                    raise ValueError("failure counts must be integers between 0 and N")
                hits = total * group[metric][f"hit_at_{cutoff}"]
                if not math.isclose(sum(counts[key] for key in FAILURES), total - hits, abs_tol=1e-6):
                    raise ValueError(f"{metric} failure counts disagree with Hit@{cutoff}")
        if sum(group["total"] for group in report["categories"].values()) != report["overall"]["total"]:
            raise ValueError("category counts do not sum to the overall count")
        if "cases" in report:
            cases = report["cases"]
            ids = [case["id"] for case in cases]
            if len(cases) != report["overall"]["total"] or len(set(ids)) != len(ids):
                raise ValueError("case IDs must be unique and case count must equal N")
            if any(type(case["evidence_hit_3"]) is not bool for case in cases):
                raise ValueError("case evidence_hit_3 values must be booleans")
            expected_hits = report["overall"]["evidence"]["hit_at_3"] * len(cases)
            if not math.isclose(sum(case["evidence_hit_3"] for case in cases), expected_hits, abs_tol=1e-6):
                raise ValueError("case hits disagree with overall evidence Hit@3")
            if cutoff == 5:
                if any(type(case["evidence_hit_5"]) is not bool for case in cases):
                    raise ValueError("case evidence_hit_5 values must be booleans")
                if not math.isclose(sum(case["evidence_hit_5"] for case in cases),
                                    report["overall"]["evidence"]["hit_at_5"] * len(cases), abs_tol=1e-6):
                    raise ValueError("case hits disagree with overall evidence Hit@5")
        return report
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid evaluation report: {error}") from error


def compatibility_issues(baseline, candidate, vary=()):
    issues = []
    for key in FIXED_CONFIG:
        if key in vary:
            continue
        left, right = baseline["config"].get(key), candidate["config"].get(key)
        if left is None or right is None:
            issues.append(f"{key}: missing; cannot verify unchanged configuration")
        elif left != right:
            issues.append(f"{key}: {left!r} → {right!r}")
    key = "documents_sha256"
    if key in baseline["config"] and key in candidate["config"]:
        if baseline["config"][key] != candidate["config"][key]:
            issues.append("documents_sha256: extracted corpus differs")
    for label, left, right in (
        ("question count", baseline["overall"]["total"], candidate["overall"]["total"]),
        ("category counts", {k: v["total"] for k, v in baseline["categories"].items()},
         {k: v["total"] for k, v in candidate["categories"].items()}),
    ):
        if left != right:
            issues.append(f"{label}: {left!r} → {right!r}")
    if "cases" in baseline and "cases" in candidate:
        left = {case["id"] for case in baseline["cases"]}
        right = {case["id"] for case in candidate["cases"]}
        if left != right:
            issues.append("case IDs: evaluated question sets differ")
    return issues


def delta(before, after, integer=False):
    if integer:
        return f"{before} → {after}  ({after - before:+d})"
    return f"{before:.3f} → {after:.3f}  ({after - before:+.3f})"


def run_label(report):
    config = report["config"]
    return (f"{config['embedding_model']} [{config.get('retrieval_method', 'dense')}; "
            f"{config.get('reranker_model', '?')}; top {config.get('rerank_top_k', 3)}]")


def compare_reports(baseline, candidate):
    lines = []
    for metric, label in METRICS.items():
        scores = dict(SCORES)
        if all("hit_at_5" in report["overall"][metric] for report in (baseline, candidate)):
            scores["hit_at_5"] = "Hit@5"
        for score, score_label in scores.items():
            lines.append(f"{label + ' ' + score_label + ':':<25} "
                         + delta(baseline["overall"][metric][score], candidate["overall"][metric][score]))
    lines.append("\nFailures (fuzzy evidence; fewer is better)")
    for failure in FAILURES:
        label = failure.replace("_failure", " failures")
        lines.append(f"{label + ':':<25} " + delta(
            baseline["overall"]["failure_counts"]["evidence"][failure],
            candidate["overall"]["failure_counts"]["evidence"][failure], integer=True))
    lines.append("\nPer category (fuzzy evidence; deltas are absolute score differences)")
    for category in sorted(baseline["categories"].keys() | candidate["categories"].keys()):
        left, right = baseline["categories"].get(category), candidate["categories"].get(category)
        if left is None or right is None:
            lines.append(f"  {category}: absent from one run; comparison unavailable")
            continue
        lines.append(f"  {category} (N: {left['total']} → {right['total']})")
        category_scores = dict(SCORES)
        if all("hit_at_5" in group["evidence"] for group in (left, right)):
            category_scores["hit_at_5"] = "Hit@5"
        for score, label in category_scores.items():
            lines.append(f"    {label + ':':<21} " + delta(left["evidence"][score], right["evidence"][score]))
        for failure in FAILURES:
            lines.append(f"    {failure.replace('_failure', ' failures') + ':':<21} " + delta(
                left["failure_counts"]["evidence"][failure], right["failure_counts"]["evidence"][failure], True))
    if "cases" in baseline and "cases" in candidate:
        left = {case["id"]: case["evidence_hit_3"] for case in baseline["cases"]}
        right = {case["id"]: case["evidence_hit_3"] for case in candidate["cases"]}
        if left.keys() == right.keys():
            gained = [key for key in left if not left[key] and right[key]]
            lost = [key for key in left if left[key] and not right[key]]
            lines.append(f"\nPaired Evidence Hit@3: {len(gained)} gained, {len(lost)} lost, "
                         f"{len(left) - len(gained) - len(lost)} unchanged")
            for label, ids in (("Gained", gained), ("Lost", lost)):
                if ids:
                    lines.append(f"{label} IDs (first 10): {', '.join(sorted(ids)[:10])}")
    else:
        lines.append("\nPaired case changes unavailable: one report contains summary data only.")
    for field, label in (("average_candidate_counts", "Mean candidate counts"),
                         ("average_latency_ms", "Mean latency (ms)")):
        if all(field in report["overall"] for report in (baseline, candidate)):
            lines.append(f"\n{label}")
            for key in sorted(baseline["overall"][field].keys() & candidate["overall"][field].keys()):
                lines.append(f"  {key}: " + delta(baseline["overall"][field][key],
                                                  candidate["overall"][field][key]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Baseline first, followed by one or more candidate JSON files")
    parser.add_argument("--allow-mismatch", action="store_true", help="Print exploratory comparisons despite differing/missing fixed settings")
    parser.add_argument("--vary", nargs="+", choices=("reranker_model",), default=[],
                        help="Declare a reranker-model experiment; all other compatibility checks remain")
    parser.add_argument("--output", type=Path, help="Also save the comparison as a text file")
    args = parser.parse_args()

    if len(args.reports) < 2:
        parser.error("provide a baseline and at least one candidate report")

    if args.output and args.output.resolve() in {Path(path).resolve() for path in args.reports}:
        parser.error("--output must not overwrite an input report")
    try:
        reports = [load_report(path) for path in args.reports]
        issues = [compatibility_issues(reports[0], report, args.vary) for report in reports[1:]]
        if any(issues) and not args.allow_mismatch:
            details = "\n".join(f"{path}:\n  " + "\n  ".join(items)
                                for path, items in zip(args.reports[1:], issues) if items)
            raise ValueError("Runs are not controlled comparisons:\n" + details
                             + "\nUse --allow-mismatch only for an explicitly exploratory comparison.")
        lines = [f"Baseline: {run_label(reports[0])} ({args.reports[0]})",
                 "Deltas are candidate minus baseline. Scores: higher is better; failures: lower is better."]
        for path, report, differences in zip(args.reports[1:], reports[1:], issues):
            lines += ["\n" + "=" * 72, f"Candidate: {run_label(report)} ({path})"]
            if differences:
                lines += ["WARNING: EXPLORATORY / MISMATCHED CONFIGURATION", *differences]
            else:
                lines.append("Fixed configuration, chunks, benchmark, and category counts match."
                             + (f" Declared experiment variables: {', '.join(args.vary)}." if args.vary else ""))
            for key in ("query_prefix", "passage_prefix", "hybrid_config", "reranker_model"):
                lines.append(f"{key}: {reports[0]['config'].get(key, '')!r} → {report['config'].get(key, '')!r}")
            lines.append(compare_reports(reports[0], report))

        if not any(issues):
            lines.append("\nRanking by Evidence Hit@3, then Evidence MRR (descriptive)")
            ranked = sorted(zip(args.reports, reports), key=lambda item: (
                item[1]["overall"]["evidence"]["hit_at_3"], item[1]["overall"]["evidence"]["mrr"]), reverse=True)
            for rank, (path, report) in enumerate(ranked, 1):
                scores = report["overall"]["evidence"]
                lines.append(f"{rank}. {run_label(report)}: "
                             f"Hit@3={scores['hit_at_3']:.3f}, MRR={scores['mrr']:.3f} ({path})")
        output = "\n".join(lines) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
