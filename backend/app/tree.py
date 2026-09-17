"""Stage 3: build the RAPTOR tree.

RAPTOR (arXiv 2401.18059) stacks layers of summaries on top of the chunks:

    embed the nodes of this level
      -> reduce dimensions (the paper uses UMAP; we use PCA, which ships with
         scikit-learn and behaves predictably on small collections)
      -> soft-cluster with a Gaussian mixture, choosing the number of clusters
         by BIC, so a node can belong to more than one cluster
      -> summarize each cluster into one parent node
      -> repeat on the parents until a single root is left

Answering then searches every level at once (the paper's "collapsed tree"),
so a detailed question can match a chunk while a broad question matches a
summary that already spans many chunks.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from . import config, llm


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    vectors = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    n_samples = vectors.shape[0]
    n_components = min(config.PCA_COMPONENTS, n_samples - 1, vectors.shape[1])
    if n_components < 2:
        return vectors
    return PCA(n_components=n_components, random_state=42).fit_transform(vectors)


def choose_cluster_count(points: np.ndarray, max_clusters: int) -> int:
    """Pick the number of clusters with the lowest BIC score."""
    best_count = 1
    best_score = float("inf")
    count = 2
    while count <= max_clusters:
        model = GaussianMixture(n_components=count, random_state=42, covariance_type="diag")
        model.fit(points)
        score = model.bic(points)
        if score < best_score:
            best_score = score
            best_count = count
        count = count + 1
    return best_count


def soft_cluster(vectors: np.ndarray) -> list[list[int]]:
    """Return, for each cluster, the row numbers that belong to it."""
    n_samples = vectors.shape[0]
    if n_samples <= 4:
        return [list(range(n_samples))]

    points = reduce_dimensions(vectors)
    max_clusters = min(config.MAX_CLUSTERS_PER_LEVEL, n_samples // 2)
    if max_clusters < 2:
        return [list(range(n_samples))]

    cluster_count = choose_cluster_count(points, max_clusters)
    if cluster_count < 2:
        return [list(range(n_samples))]

    model = GaussianMixture(n_components=cluster_count, random_state=42, covariance_type="diag")
    model.fit(points)
    probabilities = model.predict_proba(points)

    members = []
    for cluster in range(cluster_count):
        members.append([])

    for row in range(n_samples):
        joined_any = False
        for cluster in range(cluster_count):
            if probabilities[row][cluster] >= config.CLUSTER_PROB_THRESHOLD:
                members[cluster].append(row)
                joined_any = True
        if not joined_any:
            best_cluster = int(np.argmax(probabilities[row]))
            members[best_cluster].append(row)

    kept = []
    for group in members:
        if len(group) > 0:
            kept.append(group)
    return kept


def build_tree(leaves: list[dict], embedder, progress=None) -> tuple[list[dict], np.ndarray]:
    """Grow summary levels on top of the leaves.

    Returns every node (all levels) and the matching embedding matrix, in the
    same order, so `embeddings[i]` is the vector for `nodes[i]`.
    """
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    all_nodes = []
    all_vectors = []

    texts = []
    for leaf in leaves:
        texts.append(leaf["text"])
    report("embedding " + str(len(texts)) + " leaf chunks")
    level_vectors = embed_texts(embedder, texts)

    for index in range(len(leaves)):
        all_nodes.append(leaves[index])
        all_vectors.append(level_vectors[index])

    level_nodes = leaves
    level = 0
    summary_counter = 0

    while level < config.MAX_TREE_LEVELS and len(level_nodes) > 1:
        level = level + 1
        clusters = soft_cluster(level_vectors)
        report("level " + str(level) + ": " + str(len(level_nodes)) + " nodes -> "
               + str(len(clusters)) + " clusters")

        if len(clusters) >= len(level_nodes):
            # No compression happened; another level would not help.
            break

        parents = []
        for group in clusters:
            member_texts = []
            member_ids = []
            member_pages = set()
            member_sources = set()
            leaf_total = 0
            for row in group:
                node = level_nodes[row]
                member_texts.append(node["text"])
                member_ids.append(node["id"])
                for page in node["pages"]:
                    member_pages.add(page)
                member_sources.add(node["source"])
                leaf_total = leaf_total + node["member_count"]

            summary_counter = summary_counter + 1
            summary_text = llm.summarize(member_texts)

            if len(member_sources) == 1:
                source_label = list(member_sources)[0]
            else:
                source_label = str(len(member_sources)) + " documents"

            parents.append({
                "id": "S" + str(level) + "-" + str(summary_counter).zfill(3),
                "level": level,
                "kind": "summary",
                "text": summary_text,
                "source": source_label,
                "sources": sorted(member_sources),
                "pages": sorted(member_pages),
                "children": member_ids,
                "member_count": leaf_total,
            })

        parent_texts = []
        for parent in parents:
            parent_texts.append(parent["text"])
        level_vectors = embed_texts(embedder, parent_texts)

        for index in range(len(parents)):
            all_nodes.append(parents[index])
            all_vectors.append(level_vectors[index])

        level_nodes = parents
        if len(parents) == 1:
            break

    matrix = np.vstack(all_vectors).astype("float32")
    report("tree done: " + str(len(all_nodes)) + " nodes across " + str(level + 1) + " levels")
    return all_nodes, matrix


def tree_stats(nodes: list[dict]) -> dict:
    per_level = {}
    for node in nodes:
        key = str(node["level"])
        per_level[key] = per_level.get(key, 0) + 1
    sources = set()
    for node in nodes:
        if node["kind"] == "leaf":
            sources.add(node["source"])
    return {
        "nodes": len(nodes),
        "levels": len(per_level),
        "nodes_per_level": per_level,
        "documents": len(sources),
    }
