"""Background index builds, so an upload can return immediately.

A build takes seconds in fast mode and minutes with a language model, which is
far too long to hold an HTTP request open. Uploading therefore starts a job and
returns its id; the browser polls for progress and starts asking questions when
the job reports "done".

Jobs live in memory: restarting the server forgets them, but the indexes they
produced stay on disk.
"""

import threading
import time
import traceback
import uuid

from . import config, ingest, llm, store, tree

_jobs = {}
_lock = threading.Lock()


def create_job(corpus_id: str, corpus_name: str) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "corpus_id": corpus_id,
        "corpus_name": corpus_name,
        "state": "queued",          # queued | running | done | failed
        "messages": [],
        "started_at": time.time(),
        "seconds": 0.0,
        "error": None,
        "stats": None,
    }
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> dict:
    with _lock:
        return _jobs.get(job_id)


def note(job: dict, message: str) -> None:
    with _lock:
        job["messages"].append(message)
        job["seconds"] = round(time.time() - job["started_at"], 1)


def build_corpus(job: dict, raw_dir, fast: bool) -> None:
    """Chunk, embed, cluster and summarise one uploaded corpus."""
    try:
        job["state"] = "running"
        use_provider = "offline" if fast else llm.provider()
        note(job, "reading PDFs")

        leaves = ingest.build_leaves(raw_dir)
        if len(leaves) == 0:
            raise ValueError(
                "No text could be extracted. The PDF may be a scan rather than text."
            )
        note(job, "split into " + str(len(leaves)) + " chunks")

        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config.EMBED_MODEL)

        def progress(message: str) -> None:
            note(job, message)

        nodes, embeddings = tree.build_tree(
            leaves, embedder, progress=progress, use_provider=use_provider
        )

        stats = tree.tree_stats(nodes)
        meta = {
            "corpus_id": job["corpus_id"],
            "name": job["corpus_name"],
            "documents": stats["documents"],
            "leaf_chunks": len(leaves),
            "summary_provider": use_provider,
            "chunk_chars": config.CHUNK_CHARS,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "stats": stats,
            "build_seconds": round(time.time() - job["started_at"], 1),
            "summary_fallbacks": len(llm.fallbacks),
        }
        store.save_index(nodes, embeddings, meta, corpus_id=job["corpus_id"])

        job["stats"] = stats
        note(job, "done: " + str(stats["nodes"]) + " nodes across "
             + str(stats["levels"]) + " levels")
        job["state"] = "done"

    except Exception as error:
        job["error"] = str(error)
        job["state"] = "failed"
        note(job, "failed: " + str(error))
        traceback.print_exc()


def start_build(corpus_id: str, corpus_name: str, raw_dir, fast: bool) -> dict:
    job = create_job(corpus_id, corpus_name)
    thread = threading.Thread(
        target=build_corpus, args=(job, raw_dir, fast), daemon=True
    )
    thread.start()
    return job
