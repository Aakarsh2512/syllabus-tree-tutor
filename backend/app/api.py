"""Stage 6: the HTTP API.

Run it with:
    uvicorn app.api:app --reload --port 8000     (from the backend/ folder)
Then open http://127.0.0.1:8000/docs
"""

import json

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config, generate, llm, retrieve, store, tree

app = FastAPI(
    title="Syllabus Tree Tutor",
    description="RAPTOR tree-organized retrieval over your own course material.",
    version="0.1.0",
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
    top_k: int | None = None


@app.get("/api/health")
def health():
    payload = {"llm": llm.health(), "index": {"exists": store.index_exists()}}
    if store.index_exists():
        nodes, embeddings, meta = store.load_index()
        payload["index"].update({
            "meta": meta,
            "stats": tree.tree_stats(nodes),
            "embedding_dim": int(embeddings.shape[1]),
        })
    return payload


@app.get("/api/tree")
def get_tree():
    """Every node, trimmed for drawing. Leaf text is shortened to a preview."""
    nodes, _, meta = store.load_index()
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
def get_node(node_id: str):
    expanded = retrieve.expand_node(node_id)
    if expanded == {}:
        raise HTTPException(status_code=404, detail="No node with id " + node_id)
    return expanded


@app.post("/api/ask")
def ask(request: AskRequest):
    if request.mode not in ["tree", "flat"]:
        raise HTTPException(status_code=400, detail="mode must be 'tree' or 'flat'")
    return generate.answer(
        request.question, mode=request.mode, hybrid=request.hybrid,
        top_k=request.top_k, dedupe=request.dedupe,
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    if request.mode not in ["tree", "flat"]:
        raise HTTPException(status_code=400, detail="mode must be 'tree' or 'flat'")

    def event_stream():
        for event_name, payload in generate.answer_stream(
            request.question, mode=request.mode, hybrid=request.hybrid,
            top_k=request.top_k, dedupe=request.dedupe,
        ):
            body = {"event": event_name}
            body.update(payload)
            yield "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/compare")
def compare(request: AskRequest):
    return generate.compare(request.question, hybrid=request.hybrid,
                            top_k=request.top_k, dedupe=request.dedupe)


@app.get("/api/eval")
def get_eval():
    path = config.EVAL_DIR / "results.json"
    if not path.exists():
        return {"exists": False, "detail": "Run: python eval/run_eval.py"}
    with open(path, "r", encoding="utf-8") as handle:
        return {"exists": True, "results": json.load(handle)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    config.ensure_dirs()
    target = config.RAW_DIR / file.filename
    contents = await file.read()
    with open(target, "wb") as handle:
        handle.write(contents)
    return {
        "saved": file.filename,
        "bytes": len(contents),
        "next_step": "Rebuild the index: python scripts/build_index.py",
    }
