"""Find summary nodes that are not really summaries, and rewrite just those.

When an LLM call fails, `llm.summarize` falls back to extractive text: sentences
lifted verbatim out of the cluster. Those nodes are nearly useless, because a
summary node that only repeats its own children adds nothing to the tree.

Rebuilding the whole index to fix two nodes costs an hour, so this repairs them
in place: re-summarise the cluster, re-embed that one node, write it back. Any
parent built from a repaired node is repaired too, bottom-up, because its own
summary was written from the broken text.

    python scripts/repair_summaries.py --check      # report only
    python scripts/repair_summaries.py              # repair what --check finds
    python scripts/repair_summaries.py S1-001       # repair specific nodes
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import config, llm, store, tree     # noqa: E402

# A written summary paraphrases, so it shares only short runs of words with
# its children. Copied text shares long ones. Measured as the longest run of
# characters the summary and its children have in common.
#
# The threshold comes from the measurements on this corpus, which separate
# cleanly: model-written summaries share 40-65 characters with their children,
# while the node whose call failed shares 112 and two upper-level nodes that
# copied a child share 217 and 471. Anything from 100 up is a copy.
COPIED_RUN_CHARS = 100


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def longest_shared_run(summary: str, children_text: str) -> int:
    """Length of the longest stretch of text the two have word-for-word."""
    matcher = difflib.SequenceMatcher(None, summary, children_text, autojunk=False)
    match = matcher.find_longest_match(0, len(summary), 0, len(children_text))
    return match.size


def copied_run_length(node: dict, table: dict) -> int:
    summary = normalize(node["text"])
    if summary == "":
        return 0

    children_text = ""
    for child_id in node["children"]:
        if child_id in table:
            children_text = children_text + " " + normalize(table[child_id]["text"])
    if children_text == "":
        return 0

    # Comparing against every child in full is slow and unnecessary; the
    # fallback text is drawn from the start of the cluster anyway.
    return longest_shared_run(summary[:2500], children_text[:20000])


def looks_extractive(node: dict, table: dict) -> bool:
    """True when the node repeats its own children instead of summarising them.

    Catches both failure modes seen in practice: extractive fallback text after
    a failed model call, and a higher-level summary that copies one of the
    child summaries almost word for word.
    """
    return copied_run_length(node, table) >= COPIED_RUN_CHARS


def ancestors_of(node_ids: set, nodes: list[dict]) -> set:
    """Every summary node that has one of these nodes as a child, transitively."""
    affected = set(node_ids)
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node["kind"] != "summary" or node["id"] in affected:
                continue
            for child_id in node["children"]:
                if child_id in affected:
                    affected.add(node["id"])
                    changed = True
                    break
    return affected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_ids", nargs="*", help="specific nodes to repair")
    parser.add_argument("--check", action="store_true", help="report, change nothing")
    parser.add_argument("--no-cascade", action="store_true",
                        help="do not also repair parents built from repaired nodes")
    args = parser.parse_args()

    nodes, embeddings, meta = store.load_index()
    table = {}
    for node in nodes:
        table[node["id"]] = node

    if len(args.node_ids) > 0:
        broken = []
        for node_id in args.node_ids:
            if node_id not in table:
                print("no such node:", node_id)
                return
            broken.append(node_id)
    else:
        broken = []
        for node in nodes:
            if node["kind"] == "summary" and looks_extractive(node, table):
                broken.append(node["id"])

    print("summary nodes:", len([n for n in nodes if n["kind"] == "summary"]))
    print("copied from their own children:", broken if len(broken) > 0 else "none")

    if args.check:
        print()
        print("longest run of text shared with own children (>= %d = copied):"
              % COPIED_RUN_CHARS)
        for node in nodes:
            if node["kind"] != "summary":
                continue
            run = copied_run_length(node, table)
            flag = "  <-- copied" if run >= COPIED_RUN_CHARS else ""
            print("  %-8s L%d  covers %3d chunks  longest run %4d chars%s"
                  % (node["id"], node["level"], node["member_count"], run, flag))
        for node_id in broken:
            print()
            print(node_id, "|", " ".join(table[node_id]["text"].split())[:220])
        return

    if len(broken) == 0:
        print("nothing to repair")
        return

    if llm.provider() == "offline":
        print("LLM_PROVIDER is offline, so a repair would produce the same extractive text.")
        print("Set LLM_PROVIDER=ollama (or anthropic/openai) in .env first.")
        return

    targets = set(broken)
    if not args.no_cascade:
        targets = ancestors_of(targets, nodes)
        extra = sorted(targets - set(broken))
        if len(extra) > 0:
            print("also repairing parents built from them:", extra)

    # Bottom-up: a parent must be rewritten after its children.
    ordered = sorted(targets, key=lambda node_id: table[node_id]["level"])

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config.EMBED_MODEL)

    row_of = {}
    for index in range(len(nodes)):
        row_of[nodes[index]["id"]] = index

    updated = embeddings.copy()
    repaired = []
    for node_id in ordered:
        node = table[node_id]
        member_texts = []
        for child_id in node["children"]:
            if child_id in table:
                member_texts.append(table[child_id]["text"])

        print()
        print("repairing", node_id, "from", len(member_texts), "children...", flush=True)
        before = len(llm.fallbacks)
        new_text = llm.summarize(member_texts, level=node["level"])

        if len(llm.fallbacks) > before:
            print("  FAILED again, leaving the old text in place")
            continue

        node["text"] = new_text
        vector = embedder.encode([new_text], normalize_embeddings=True)
        updated[row_of[node_id]] = np.asarray(vector, dtype="float32")[0]
        repaired.append(node_id)
        print("  ok:", " ".join(new_text.split())[:180])

    if len(repaired) == 0:
        print()
        print("nothing was repaired")
        return

    meta = dict(meta)
    meta["repaired_nodes"] = repaired
    meta["summary_fallbacks"] = max(0, meta.get("summary_fallbacks", 0) - len(repaired))
    store.save_index(nodes, updated, meta)

    print()
    print("repaired", len(repaired), "nodes:", repaired)
    print("index rewritten. Re-run the evaluation to see the effect.")
    print("stats:", tree.tree_stats(nodes)["nodes_per_level"])


if __name__ == "__main__":
    main()
