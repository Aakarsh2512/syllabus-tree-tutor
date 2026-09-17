"""Stage 8: score the flat baseline against the RAPTOR tree on the same questions.

    python eval/run_eval.py
    python eval/run_eval.py --top-k 6 --runs flat tree tree_hybrid

Metrics
-------
specific questions   hit rate @k  - did any retrieved node contain the evidence?
                     MRR          - 1 / rank of the first node that did
broad questions      coverage @k  - what fraction of the several required
                                    pieces of evidence the retrieved set holds
unanswerable         refusal rate - did the answer decline instead of inventing

Writes eval/results.json (read by the web app) and eval/results.md.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import config, generate, llm     # noqa: E402

REFUSAL_PATTERNS = ["couldn't find", "could not find", "not in the provided", "no information"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_golden() -> list[dict]:
    path = ROOT / "eval" / "golden.jsonl"
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                items.append(json.loads(line))
    return items


def find_evidence_rank(hits: list[dict], evidence: str) -> int:
    """1-based rank of the first retrieved node containing the evidence, else 0."""
    needle = normalize(evidence)
    for hit in hits:
        if needle in normalize(hit["text"]):
            return hit["rank"]
    return 0


def score_item(item: dict, result: dict) -> dict:
    hits = result["hits"]
    category = item["category"]
    row = {
        "id": item["id"],
        "category": category,
        "question": item["question"],
        "retrieved": result["trace"]["retrieved_ids"],
        "summary_hits": result["trace"]["summary_hits"],
        "levels_used": result["trace"]["levels_used"],
    }

    if category == "specific":
        best_rank = 0
        for evidence in item["evidence_any"]:
            rank = find_evidence_rank(hits, evidence)
            if rank > 0 and (best_rank == 0 or rank < best_rank):
                best_rank = rank
        row["hit"] = 1 if best_rank > 0 else 0
        row["rank"] = best_rank
        row["reciprocal_rank"] = (1.0 / best_rank) if best_rank > 0 else 0.0

    elif category == "broad":
        found = 0
        missing = []
        for evidence in item["evidence_all"]:
            if find_evidence_rank(hits, evidence) > 0:
                found = found + 1
            else:
                missing.append(evidence)
        row["coverage"] = found / len(item["evidence_all"])
        row["missing"] = missing

    else:
        answer_text = normalize(result["answer"])
        refused = False
        for pattern in REFUSAL_PATTERNS:
            if pattern in answer_text:
                refused = True
                break
        row["refused"] = 1 if refused else 0

    return row


def average(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return round(sum(values) / len(values), 4)


def summarize_rows(rows: list[dict]) -> dict:
    specific = [row for row in rows if row["category"] == "specific"]
    broad = [row for row in rows if row["category"] == "broad"]
    unanswerable = [row for row in rows if row["category"] == "unanswerable"]

    return {
        "specific_hit_rate": average([row["hit"] for row in specific]),
        "specific_mrr": average([row["reciprocal_rank"] for row in specific]),
        "broad_coverage": average([row["coverage"] for row in broad]),
        "refusal_rate": average([row["refused"] for row in unanswerable]),
        "avg_summary_hits": average([row["summary_hits"] for row in rows]),
        "counts": {
            "specific": len(specific),
            "broad": len(broad),
            "unanswerable": len(unanswerable),
        },
    }


RUN_SETTINGS = {
    "flat": {"mode": "flat", "hybrid": False, "dedupe": True},
    "tree": {"mode": "tree", "hybrid": False, "dedupe": True},
    "tree_hybrid": {"mode": "tree", "hybrid": True, "dedupe": True},
    "tree_no_dedupe": {"mode": "tree", "hybrid": False, "dedupe": False},
    "flat_no_dedupe": {"mode": "flat", "hybrid": False, "dedupe": False},
}


def run_one(name: str, golden: list[dict], top_k: int) -> dict:
    settings = RUN_SETTINGS[name]
    print("running", name, settings)
    rows = []
    started = time.time()
    for item in golden:
        result = generate.answer(
            item["question"],
            mode=settings["mode"],
            hybrid=settings["hybrid"],
            dedupe=settings["dedupe"],
            top_k=top_k,
        )
        rows.append(score_item(item, result))
        print("  .", end="", flush=True)
    print()
    elapsed = time.time() - started

    return {
        "name": name,
        "settings": settings,
        "top_k": top_k,
        "metrics": summarize_rows(rows),
        "seconds_total": round(elapsed, 1),
        "seconds_per_question": round(elapsed / max(1, len(golden)), 2),
        "rows": rows,
    }


def write_markdown(results: dict, path: Path) -> None:
    lines = []
    lines.append("# Evaluation results")
    lines.append("")
    lines.append("Corpus: " + str(results["index"]["documents"]) + " documents, "
                 + str(results["index"]["leaf_chunks"]) + " leaf chunks, "
                 + str(results["index"]["stats"]["nodes"]) + " tree nodes across "
                 + str(results["index"]["stats"]["levels"]) + " levels.")
    lines.append("")
    lines.append("Questions: " + json.dumps(results["runs"][0]["metrics"]["counts"]) + ". ")
    lines.append("Answer provider: `" + results["provider"] + "`. top_k = "
                 + str(results["runs"][0]["top_k"]) + ".")
    lines.append("")
    lines.append("| Run | Specific hit@k | Specific MRR | Broad coverage | Refusal rate | Summary nodes / query | s per query |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in results["runs"]:
        metrics = run["metrics"]
        lines.append("| `" + run["name"] + "` | "
                     + f"{metrics['specific_hit_rate']:.3f} | "
                     + f"{metrics['specific_mrr']:.3f} | "
                     + f"{metrics['broad_coverage']:.3f} | "
                     + f"{metrics['refusal_rate']:.3f} | "
                     + f"{metrics['avg_summary_hits']:.2f} | "
                     + f"{run['seconds_per_question']:.2f} |")
    lines.append("")
    lines.append("Metric meanings are in `eval/run_eval.py`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--runs", nargs="+", default=["flat", "tree", "tree_hybrid"])
    args = parser.parse_args()

    for name in args.runs:
        if name not in RUN_SETTINGS:
            print("unknown run:", name, "| choose from:", list(RUN_SETTINGS))
            return

    golden = load_golden()
    print("golden questions:", len(golden), "| provider:", llm.provider())

    # Load the embedding model before timing, so the first run is not charged
    # for a one-off model load.
    from app import retrieve
    retrieve.embed_query("warm up")

    from app import store, tree as tree_module
    nodes, _, meta = store.load_index()

    runs = []
    for name in args.runs:
        runs.append(run_one(name, golden, args.top_k))

    results = {
        "provider": llm.provider(),
        "index": {
            "documents": meta.get("documents"),
            "document_names": meta.get("document_names", []),
            "leaf_chunks": meta.get("leaf_chunks"),
            "stats": tree_module.tree_stats(nodes),
            "summary_provider": meta.get("summary_provider"),
        },
        "runs": runs,
    }

    out_json = ROOT / "eval" / "results.json"
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    write_markdown(results, ROOT / "eval" / "results.md")

    print()
    header = f"{'run':16} {'spec hit@k':>11} {'spec MRR':>9} {'broad cov':>10} {'refusal':>8} {'summ/q':>7}"
    print(header)
    print("-" * len(header))
    for run in runs:
        metrics = run["metrics"]
        print(f"{run['name']:16} {metrics['specific_hit_rate']:>11.3f} "
              f"{metrics['specific_mrr']:>9.3f} {metrics['broad_coverage']:>10.3f} "
              f"{metrics['refusal_rate']:>8.3f} {metrics['avg_summary_hits']:>7.2f}")
    print()
    print("wrote", out_json.name, "and results.md")


if __name__ == "__main__":
    main()
