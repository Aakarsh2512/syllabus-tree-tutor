"""Saving and loading the index, and one cached copy for the API to use."""

import json
from datetime import datetime, timezone

import numpy as np

from . import config

_cache = {"nodes": None, "embeddings": None, "meta": None}


def save_index(nodes: list[dict], embeddings: np.ndarray, meta: dict) -> None:
    config.ensure_dirs()
    with open(config.NODES_PATH, "w", encoding="utf-8") as handle:
        json.dump(nodes, handle, ensure_ascii=False, indent=1)
    np.save(config.EMBEDDINGS_PATH, embeddings)

    meta = dict(meta)
    meta["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["embed_model"] = config.EMBED_MODEL
    with open(config.META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)

    _cache["nodes"] = None
    _cache["embeddings"] = None
    _cache["meta"] = None


def index_exists() -> bool:
    return config.NODES_PATH.exists() and config.EMBEDDINGS_PATH.exists()


def load_index() -> tuple[list[dict], np.ndarray, dict]:
    if _cache["nodes"] is None:
        if not index_exists():
            raise FileNotFoundError(
                "No index found. Run: python scripts/build_index.py"
            )
        with open(config.NODES_PATH, "r", encoding="utf-8") as handle:
            _cache["nodes"] = json.load(handle)
        _cache["embeddings"] = np.load(config.EMBEDDINGS_PATH)
        meta = {}
        if config.META_PATH.exists():
            with open(config.META_PATH, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        _cache["meta"] = meta
    return _cache["nodes"], _cache["embeddings"], _cache["meta"]


def nodes_by_id() -> dict:
    nodes, _, _ = load_index()
    table = {}
    for node in nodes:
        table[node["id"]] = node
    return table
