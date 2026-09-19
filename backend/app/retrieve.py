"""Stage 4: two retrievers over one index.

    mode="flat"  -> search only the leaf chunks. This is the baseline, and it
                    is what an ordinary RAG system does.
    mode="tree"  -> search every node at every level at once: chunks and
                    summaries together. This is RAPTOR's "collapsed tree".

Both read the same text, so a difference in answers comes from the tree, not
from different chunking or a different embedding model.
"""

import numpy as np

from . import config, store

_embedder = {"model": None}


def get_embedder():
    if _embedder["model"] is None:
        from sentence_transformers import SentenceTransformer

        _embedder["model"] = SentenceTransformer(config.EMBED_MODEL)
    return _embedder["model"]


def embed_query(question: str) -> np.ndarray:
    vector = get_embedder().encode([question], normalize_embeddings=True)
    return np.asarray(vector, dtype="float32")[0]


def bm25_scores(question: str, texts: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    corpus = []
    for text in texts:
        corpus.append(text.lower().split())
    model = BM25Okapi(corpus)
    return np.asarray(model.get_scores(question.lower().split()), dtype="float32")


def fuse_ranks(dense_order: list[int], sparse_order: list[int], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merge two rankings using positions, not scores."""
    fused = {}
    for ranking in [dense_order, sparse_order]:
        rank = 1
        for row in ranking:
            if row not in fused:
                fused[row] = 0.0
            fused[row] = fused[row] + 1.0 / (k + rank)
            rank = rank + 1
    return sorted(fused, key=fused.get, reverse=True)


def drop_near_duplicates(order: list[int], matrix: np.ndarray, k: int, threshold: float):
    """Keep the best results, skipping any that repeat one already kept.

    The demo corpus contains four variants of the same quiz, so without this
    the top 6 results were four copies of the same page. Vectors are
    normalized, so a dot product is the cosine similarity.
    """
    kept = []
    dropped = []
    for row in order:
        is_duplicate = False
        for kept_row in kept:
            if float(matrix[row] @ matrix[kept_row]) >= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            dropped.append(row)
        else:
            kept.append(row)
        if len(kept) >= k:
            break
    return kept, dropped


def take_slots(order: list[int], candidate_rows: list[int], nodes: list[dict],
               matrix: np.ndarray, k: int, threshold: float, dedupe: bool,
               max_summaries: int):
    """Fill k result slots in ranked order, with two limits.

    Near-duplicates are skipped, and at most `max_summaries` of the slots may
    go to summary nodes. The cap exists because summaries compete with chunks
    for a fixed number of slots: on a broad question a summary is worth more
    than the chunk it displaces, but on a question whose answer sits in three
    specific chunks, three summaries in the top six is a straight loss.
    """
    kept = []
    dropped = []
    summaries_taken = 0
    for row in order:
        is_summary = nodes[candidate_rows[row]]["kind"] == "summary"
        if max_summaries is not None and is_summary and summaries_taken >= max_summaries:
            continue

        if dedupe:
            is_duplicate = False
            for kept_row in kept:
                if float(matrix[row] @ matrix[kept_row]) >= threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                dropped.append(row)
                continue

        kept.append(row)
        if is_summary:
            summaries_taken = summaries_taken + 1
        if len(kept) >= k:
            break
    return kept, dropped


def search(question: str, mode: str = "tree", top_k: int = None, hybrid: bool = False,
           dedupe: bool = True, summary_penalty: float = None,
           max_summaries: int = None, corpus_id: str = None) -> dict:
    """Return the top nodes for a question, plus a trace of how they were found."""
    nodes, embeddings, meta = store.load_index(corpus_id)
    k = top_k if top_k is not None else config.TOP_K

    candidate_rows = []
    for row in range(len(nodes)):
        if mode == "flat" and nodes[row]["kind"] != "leaf":
            continue
        candidate_rows.append(row)

    question_vector = embed_query(question)
    candidate_matrix = embeddings[candidate_rows]
    dense = candidate_matrix @ question_vector          # cosine: vectors are normalized

    penalty = config.SUMMARY_PENALTY if summary_penalty is None else summary_penalty
    ranking_scores = dense.copy()
    if penalty != 0.0:
        for local_row in range(len(candidate_rows)):
            if nodes[candidate_rows[local_row]]["kind"] == "summary":
                ranking_scores[local_row] = ranking_scores[local_row] - penalty

    dense_order_local = np.argsort(ranking_scores)[::-1].tolist()

    if hybrid:
        texts = []
        for row in candidate_rows:
            texts.append(nodes[row]["text"])
        sparse = bm25_scores(question, texts)
        sparse_order_local = np.argsort(sparse)[::-1].tolist()
        order_local = fuse_ranks(dense_order_local[:50], sparse_order_local[:50])
    else:
        order_local = dense_order_local

    chosen_local, dropped_local = take_slots(
        order_local, candidate_rows, nodes, candidate_matrix, k,
        config.DEDUPE_THRESHOLD, dedupe, max_summaries,
    )

    hits = []
    rank = 1
    for local_row in chosen_local:
        row = candidate_rows[local_row]
        node = nodes[row]
        hits.append({
            "rank": rank,
            "id": node["id"],
            "level": node["level"],
            "kind": node["kind"],
            "source": node["source"],
            "pages": node["pages"],
            "member_count": node["member_count"],
            "score": round(float(dense[local_row]), 4),
            "text": node["text"],
            "children": node["children"],
        })
        rank = rank + 1

    levels_used = {}
    for hit in hits:
        key = str(hit["level"])
        levels_used[key] = levels_used.get(key, 0) + 1

    summary_hits = 0
    for hit in hits:
        if hit["kind"] == "summary":
            summary_hits = summary_hits + 1

    duplicate_ids = []
    for local_row in dropped_local:
        duplicate_ids.append(nodes[candidate_rows[local_row]]["id"])

    trace = {
        "mode": mode,
        "hybrid": hybrid,
        "dedupe": dedupe,
        "summary_penalty": penalty,
        "max_summaries": max_summaries,
        "duplicates_skipped": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "top_k": k,
        "searched_nodes": len(candidate_rows),
        "total_nodes": len(nodes),
        "levels_used": levels_used,
        "summary_hits": summary_hits,
        "leaf_hits": len(hits) - summary_hits,
        "retrieved_ids": [hit["id"] for hit in hits],
        "top_score": hits[0]["score"] if len(hits) > 0 else 0.0,
    }
    return {"hits": hits, "trace": trace, "index_meta": meta}


def expand_node(node_id: str, corpus_id: str = None) -> dict:
    """A summary node plus the nodes it was built from, for the UI."""
    table = store.nodes_by_id(corpus_id)
    if node_id not in table:
        return {}
    node = table[node_id]
    children = []
    for child_id in node["children"]:
        if child_id in table:
            child = table[child_id]
            children.append({
                "id": child["id"],
                "level": child["level"],
                "kind": child["kind"],
                "source": child["source"],
                "pages": child["pages"],
                "text": child["text"],
            })
    return {"node": node, "children": children}
