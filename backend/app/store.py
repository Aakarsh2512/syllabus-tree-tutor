"""Saving and loading the index, and one cached copy for the API to use."""

import json
from datetime import datetime, timezone

import numpy as np

from . import config

# One entry per corpus, so switching between uploads does not reload from disk.
_cache = {}


def corpus_dir(corpus_id: str = None):
    """Where one corpus keeps its index. The demo corpus keeps the old path."""
    if corpus_id is None or corpus_id == config.DEMO_CORPUS_ID:
        return config.INDEX_DIR
    return config.CORPORA_DIR / corpus_id


def corpus_raw_dir(corpus_id: str = None):
    if corpus_id is None or corpus_id == config.DEMO_CORPUS_ID:
        return config.RAW_DIR
    return config.CORPORA_DIR / corpus_id / "raw"


def paths_for(corpus_id: str = None) -> tuple:
    directory = corpus_dir(corpus_id)
    return (directory / "nodes.json", directory / "embeddings.npy",
            directory / "meta.json")


def save_index(nodes: list[dict], embeddings: np.ndarray, meta: dict,
               corpus_id: str = None) -> None:
    config.ensure_dirs()
    nodes_path, embeddings_path, meta_path = paths_for(corpus_id)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(nodes_path, "w", encoding="utf-8") as handle:
        json.dump(nodes, handle, ensure_ascii=False, indent=1)
    np.save(embeddings_path, embeddings)

    meta = dict(meta)
    meta["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["embed_model"] = config.EMBED_MODEL
    meta.setdefault("corpus_id", corpus_id or config.DEMO_CORPUS_ID)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)

    _cache.pop(corpus_id or config.DEMO_CORPUS_ID, None)


def index_exists(corpus_id: str = None) -> bool:
    nodes_path, embeddings_path, _ = paths_for(corpus_id)
    return nodes_path.exists() and embeddings_path.exists()


def load_index(corpus_id: str = None) -> tuple[list[dict], np.ndarray, dict]:
    key = corpus_id or config.DEMO_CORPUS_ID
    if key not in _cache:
        if not index_exists(corpus_id):
            raise FileNotFoundError(
                "No index for corpus " + key + ". Upload a PDF, or run: "
                "python scripts/build_index.py"
            )
        nodes_path, embeddings_path, meta_path = paths_for(corpus_id)
        with open(nodes_path, "r", encoding="utf-8") as handle:
            nodes = json.load(handle)
        embeddings = np.load(embeddings_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        _cache[key] = (nodes, embeddings, meta)
    return _cache[key]


def list_corpora() -> list[dict]:
    """Every index on disk: the demo one, plus anything uploaded."""
    found = []
    if index_exists(None):
        _, _, meta = load_index(None)
        found.append({
            "corpus_id": config.DEMO_CORPUS_ID,
            "name": meta.get("name", "Demo: course material"),
            "built_at": meta.get("built_at"),
            "stats": meta.get("stats"),
            "documents": meta.get("document_names", []),
            "summary_provider": meta.get("summary_provider"),
        })

    if config.CORPORA_DIR.exists():
        for directory in sorted(config.CORPORA_DIR.iterdir()):
            if not directory.is_dir():
                continue
            corpus_id = directory.name
            if not index_exists(corpus_id):
                continue
            _, _, meta = load_index(corpus_id)
            found.append({
                "corpus_id": corpus_id,
                "name": meta.get("name", corpus_id),
                "built_at": meta.get("built_at"),
                "stats": meta.get("stats"),
                "documents": meta.get("document_names", []),
                "summary_provider": meta.get("summary_provider"),
            })
    return found


def nodes_by_id(corpus_id: str = None) -> dict:
    nodes, _, _ = load_index(corpus_id)
    table = {}
    for node in nodes:
        table[node["id"]] = node
    return table
