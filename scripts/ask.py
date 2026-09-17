"""Ask a question from the terminal, without the web app.

Usage:
    python scripts/ask.py "what is slip in an induction motor?"
    python scripts/ask.py --mode flat "..."
    python scripts/ask.py --compare "summarise what the quiz covers"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# The Windows console is cp1252 by default and course notes are full of
# characters like the minus sign and multiplication sign.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import generate, llm     # noqa: E402


def show(result: dict, title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(result["answer"])
    print()
    print("retrieved:", result["trace"]["retrieved_ids"])
    print("levels used:", result["trace"]["levels_used"],
          "| summaries:", result["trace"]["summary_hits"],
          "| leaves:", result["trace"]["leaf_hits"],
          "| top score:", result["trace"]["top_score"])
    print("citations:", result["citations"])
    print("duplicates skipped:", result["trace"]["duplicates_skipped"],
          result["trace"]["duplicate_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", default="tree", choices=["tree", "flat"])
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    print("provider:", llm.provider())

    if args.compare:
        both = generate.compare(args.question, hybrid=args.hybrid, top_k=args.top_k,
                                dedupe=not args.no_dedupe)
        show(both["flat"], "FLAT baseline (leaf chunks only)")
        show(both["tree"], "RAPTOR tree (all levels)")
        print()
        print("nodes retrieved by both:", both["shared_count"], both["shared_nodes"])
        return

    result = generate.answer(
        args.question, mode=args.mode, hybrid=args.hybrid, top_k=args.top_k,
        dedupe=not args.no_dedupe,
    )
    show(result, args.mode.upper() + " mode")


if __name__ == "__main__":
    main()
