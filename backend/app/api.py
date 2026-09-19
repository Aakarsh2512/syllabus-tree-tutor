"""The HTTP API.

Run it with:
    uvicorn app.api:app --reload --port 8000     (from the backend/ folder)
Then open http://127.0.0.1:8000/docs

Every question endpoint takes an optional `corpus_id`. Leaving it out uses the
demo corpus; uploading PDFs creates a new one.
"""

import json
import re
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config, generate, jobs, llm, retrieve, store, tree

app = FastAPI(
    title="Syllabus Tree Tutor",
    description="RAPTOR tree-organized retrieval over your own PDFs, "
                "with a flat-chunk baseline to compare against.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: str = "tree"                 # "tree" or "flat"
    hybrid: bool = False
    dedupe: bool = True
    top_k: int | None = Field(default=None, ge=1, le=30)
    corpus_id: str | None = None
    # Answer without a language model: instant, extractive, and enough to
    # see which passages each retriever found.
    fast_answers: bool = False


def safe_filename(name: str) -> str:
    """Strip any directory parts, so an upload cannot write outside its folder."""
    base = name.replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip()
    if cleaned == "" or cleaned == ".pdf":
        cleaned = "document.pdf"
    return cleaned[:120]


def require_index(corpus_id: str | None):
    if not store.index_exists(corpus_id):
        raise HTTPException(
            status_code=404,
            detail="No index for that corpus yet. Upload a PDF first.",
        )


# ------------------------------------------------------------------ status --

@app.get("/api/health")
def health():
    payload = {
        "llm": llm.health(),
        "corpora": store.list_corpora(),
        "index": {"exists": store.index_exists(None)},
    }
    if store.index_exists(None):
        nodes, embeddings, meta = store.load_index(None)
        payload["index"].update({
            "meta": meta,
            "stats": tree.tree_stats(nodes),
            "embedding_dim": int(embeddings.shape[1]),
        })
    return payload


@app.get("/api/corpora")
def corpora():
    return {"corpora": store.list_corpora(), "demo_id": config.DEMO_CORPUS_ID}


# ------------------------------------------------------------------ upload --

@app.post("/api/corpus/upload")
async def upload_corpus(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    fast: bool = Form(True),
):
    """Take PDFs, start building their tree, and return a job to watch.

    `fast=true` writes extractive summaries, which takes seconds. `fast=false`
    uses the configured language model, which is better but can take minutes
    per summary on a CPU.
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files received")
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail="At most " + str(config.MAX_UPLOAD_FILES) + " files at a time",
        )

    corpus_id = uuid.uuid4().hex[:10]
    raw_dir = store.corpus_raw_dir(corpus_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    total_bytes = 0
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted: " + upload.filename,
            )
        contents = await upload.read()
        total_bytes = total_bytes + len(contents)
        if total_bytes > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Upload is larger than "
                       + str(config.MAX_UPLOAD_BYTES // (1024 * 1024)) + " MB",
            )
        target = raw_dir / safe_filename(upload.filename)
        with open(target, "wb") as handle:
            handle.write(contents)
        saved.append(target.name)

    corpus_name = name.strip()
    if corpus_name == "":
        corpus_name = saved[0]
        if len(saved) > 1:
            corpus_name = corpus_name + " and " + str(len(saved) - 1) + " more"

    job = jobs.start_build(corpus_id, corpus_name, raw_dir, fast)
    return {
        "corpus_id": corpus_id,
        "job_id": job["id"],
        "name": corpus_name,
        "files": saved,
        "bytes": total_bytes,
        "summaries_by": "offline (fast)" if fast else llm.provider(),
    }


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return job


# ------------------------------------------------------------------- index --

@app.get("/api/tree")
def get_tree(corpus_id: str | None = None):
    """Every node, trimmed for drawing. Leaf text is shortened to a preview."""
    require_index(corpus_id)
    nodes, _, meta = store.load_index(corpus_id)
    trimmed = []
    for node in nodes:
        preview = node["text"]
        if node["kind"] == "leaf" and len(preview) > 280:
            preview = preview[:280] + "..."
        trimmed.append({
            "id": node["id"],
            "level": node["level"],
            "kind": node["kind"],
            "source": node["source"],
            "pages": node["pages"],
            "member_count": node["member_count"],
            "children": node["children"],
            "preview": preview,
        })
    return {"nodes": trimmed, "meta": meta, "stats": tree.tree_stats(nodes)}


@app.get("/api/node/{node_id}")
def get_node(node_id: str, corpus_id: str | None = None):
    require_index(corpus_id)
    expanded = retrieve.expand_node(node_id, corpus_id=corpus_id)
    if expanded == {}:
        raise HTTPException(status_code=404, detail="No node with id " + node_id)
    return expanded


# ---------------------------------------------------------------- answering --

def answer_provider_for(request: AskRequest):
    return "offline" if request.fast_answers else None


@app.post("/api/ask")
def ask(request: AskRequest):
    if request.mode not in ["tree", "flat"]:
        raise HTTPException(status_code=400, detail="mode must be 'tree' or 'flat'")
    require_index(request.corpus_id)
    return generate.answer(
        request.question, mode=request.mode, hybrid=request.hybrid,
        top_k=request.top_k, dedupe=request.dedupe, corpus_id=request.corpus_id,
        answer_provider=answer_provider_for(request),
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    if request.mode not in ["tree", "flat"]:
        raise HTTPException(status_code=400, detail="mode must be 'tree' or 'flat'")
    require_index(request.corpus_id)

    def event_stream():
        for event_name, payload in generate.answer_stream(
            request.question, mode=request.mode, hybrid=request.hybrid,
            top_k=request.top_k, dedupe=request.dedupe, corpus_id=request.corpus_id,
            answer_provider=answer_provider_for(request),
        ):
            body = {"event": event_name}
            body.update(payload)
            yield "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/compare")
def compare(request: AskRequest):
    require_index(request.corpus_id)
    return generate.compare(
        request.question, hybrid=request.hybrid, top_k=request.top_k,
        dedupe=request.dedupe, corpus_id=request.corpus_id,
        answer_provider=answer_provider_for(request),
    )


@app.post("/api/compare/stream")
def compare_stream(request: AskRequest):
    """Both retrievers on one question, streamed: passages first, answers after."""
    require_index(request.corpus_id)

    def event_stream():
        for event_name, payload in generate.compare_stream(
            request.question, hybrid=request.hybrid, top_k=request.top_k,
            dedupe=request.dedupe, corpus_id=request.corpus_id,
            answer_provider=answer_provider_for(request),
        ):
            body = {"event": event_name}
            body.update(payload)
            yield "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --------------------------------------------------------------- evaluation --

@app.get("/api/eval")
def get_eval():
    path = config.EVAL_DIR / "results.json"
    if not path.exists():
        return {"exists": False, "detail": "Run: python eval/run_eval.py"}
    with open(path, "r", encoding="utf-8") as handle:
        return {"exists": True, "results": json.load(handle)}
