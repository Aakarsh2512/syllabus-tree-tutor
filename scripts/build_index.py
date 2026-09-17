"""Build the index: data/raw/*.pdf  ->  data/index/

Usage:
    python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# The Windows console is cp1252 by default and course notes are full of
# characters like the minus sign and multiplication sign.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import config, ingest, llm, store, tree     # noqa: E402


def main() -> None:
    config.ensure_dirs()

    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if len(pdfs) == 0:
        print("No PDFs in " + str(config.RAW_DIR) + ". Put some course material there first.")
        return

    print("provider:", llm.provider(), "|", llm.health()["detail"])
    print("documents:", len(pdfs))
    for path in pdfs:
        print("  -", path.name)

    started = time.time()
    leaves = ingest.build_leaves()
    print("leaf chunks:", len(leaves))
    if len(leaves) == 0:
        print("No text could be extracted. Are these scanned PDFs?")
        return

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config.EMBED_MODEL)

    def progress(message: str) -> None:
        print("  " + message)

    nodes, embeddings = tree.build_tree(leaves, embedder, progress=progress)

    stats = tree.tree_stats(nodes)
    meta = {
        "documents": len(pdfs),
        "document_names": [path.name for path in pdfs],
        "leaf_chunks": len(leaves),
        "summary_provider": llm.provider(),
        "chunk_chars": config.CHUNK_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "stats": stats,
        "build_seconds": round(time.time() - started, 1),
    }
    store.save_index(nodes, embeddings, meta)

    print()
    print("saved to", config.INDEX_DIR)
    print("nodes:", stats["nodes"], "| levels:", stats["levels"], "| per level:", stats["nodes_per_level"])
    print("took", meta["build_seconds"], "seconds")


if __name__ == "__main__":
    main()
