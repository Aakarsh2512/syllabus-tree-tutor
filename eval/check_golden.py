"""Check that the broad questions can actually tell the two retrievers apart.

Version 1 of the question set could not: `flat` and `tree` both scored 0.938,
because six chunks happened to contain all the evidence, so the metric was
pinned at its ceiling and no difference could show either way.

A broad question only discriminates when its evidence is spread across more
chunks than the retriever has slots. This measures that spread, for each
question, from the corpus alone — it never runs either retriever, so the set
cannot be tuned towards a winner.

    min_leaves_to_cover  fewest chunks that together hold all the evidence.
                         Must be >= MIN_SPREAD, or a single lucky chunk wins.
    max_in_one_leaf      most evidence terms found in any single chunk.
                         If this equals the number of terms, the question is
                         secretly a specific question.

    python eval/check_golden.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import store     # noqa: E402

MIN_SPREAD = 3
TOP_K = 6


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_golden(name: str) -> list[dict]:
    items = []
    with open(ROOT / "eval" / name, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                items.append(json.loads(line))
    return items


def spread_of(terms: list[str], leaves: list[tuple]) -> dict:
    """How widely the evidence for one question is scattered over the chunks."""
    missing = []
    for term in terms:
        found = False
        for _, text in leaves:
            if normalize(term) in text:
                found = True
                break
        if not found:
            missing.append(term)

    max_in_one = 0
    for _, text in leaves:
        count = 0
        for term in terms:
            if normalize(term) in text:
                count = count + 1
        if count > max_in_one:
            max_in_one = count

    remaining = set(terms) - set(missing)
    used = 0
    while len(remaining) > 0:
        best_text = ""
        best_gain = 0
        for _, text in leaves:
            gain = 0
            for term in remaining:
                if normalize(term) in text:
                    gain = gain + 1
            if gain > best_gain:
                best_gain = gain
                best_text = text
        if best_gain == 0:
            break
        used = used + 1
        still = set()
        for term in remaining:
            if normalize(term) not in best_text:
                still.add(term)
        remaining = still

    return {"missing": missing, "max_in_one_leaf": max_in_one, "min_leaves_to_cover": used}


def summary_coverage(terms: list[str], summaries: list[tuple]) -> tuple:
    best_id = "-"
    best_count = 0
    for node_id, text in summaries:
        count = 0
        for term in terms:
            if normalize(term) in text:
                count = count + 1
        if count > best_count:
            best_count = count
            best_id = node_id
    return best_id, best_count


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "golden.jsonl"
    nodes, _, _ = store.load_index()

    leaves = []
    summaries = []
    for node in nodes:
        if node["kind"] == "leaf":
            leaves.append((node["id"], normalize(node["text"])))
        else:
            summaries.append((node["id"], normalize(node["text"])))

    golden = load_golden(name)
    print(name, "|", len(golden), "questions |", len(leaves), "chunks,",
          len(summaries), "summaries | top_k =", TOP_K)
    print()
    header = "%-5s %6s %9s %11s   %-22s %s" % (
        "id", "terms", "max/leaf", "min leaves", "best summary", "verdict")
    print(header)
    print("-" * len(header))

    problems = []
    for item in golden:
        if item["category"] != "broad":
            continue
        terms = item["evidence_all"]
        spread = spread_of(terms, leaves)
        best_id, best_count = summary_coverage(terms, summaries)

        verdict = "ok"
        if len(spread["missing"]) > 0:
            verdict = "MISSING from corpus: " + ", ".join(spread["missing"])
            problems.append(item["id"])
        elif spread["max_in_one_leaf"] >= len(terms):
            verdict = "not broad: one chunk holds every term"
            problems.append(item["id"])
        elif spread["min_leaves_to_cover"] < MIN_SPREAD:
            if item.get("counter_case") is True:
                verdict = "low spread, kept on purpose (counter-case)"
            else:
                verdict = "too concentrated: needs >= %d chunks" % MIN_SPREAD
                problems.append(item["id"])
        elif best_count == 0:
            verdict = "ok - no summary helps here (counter-case)"

        print("%-5s %6d %9d %11d   %-22s %s" % (
            item["id"], len(terms), spread["max_in_one_leaf"],
            spread["min_leaves_to_cover"], best_id + " covers " + str(best_count),
            verdict))

    print()
    if len(problems) == 0:
        print("All broad questions are spread across at least", MIN_SPREAD,
              "chunks, so the metric has room to show a difference.")
    else:
        print("FIX THESE:", ", ".join(problems))


if __name__ == "__main__":
    main()
