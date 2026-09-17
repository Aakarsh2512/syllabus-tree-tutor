# Syllabus Tree Tutor

Question answering over your own course material, built on **RAPTOR: Recursive
Abstractive Processing for Tree-Organized Retrieval** ([arXiv 2401.18059](https://arxiv.org/abs/2401.18059)).

Ordinary RAG splits documents into chunks and searches the chunks. That works for
narrow questions ("what is the full-load slip?") and poorly for broad ones
("what does this quiz cover?"), because no single chunk contains the answer to a
broad question. RAPTOR builds **layers of summaries** on top of the chunks and
searches every layer at once, so a broad question can match a summary that
already spans thirty chunks.

This project implements that, and — more importantly — **measures whether it
actually helps**, against a flat-chunk baseline on the same index.

![The RAPTOR tree](docs/tree.png)

## What it does

- **Builds a tree** from a folder of PDFs: chunk, embed, cluster, summarise, repeat to a single root.
- **Answers questions** with inline citations back to the source file and page, streamed token by token.
- **Shows its working**: which nodes were retrieved, from which level, their cosine scores, and how many near-duplicates were skipped.
- **Compares itself to the baseline** side by side on the same question.
- **Scores itself** on a hand-written 26-question evaluation set.
- **Runs with no API key at all** (extractive fallback), or with a local model via Ollama, or with Claude/OpenAI.

## How the tree is built

```
192 leaf chunks
      ↓  embed (all-MiniLM-L6-v2, 384 dims)
      ↓  PCA to 10 dims
      ↓  Gaussian mixture, cluster count chosen by BIC, soft assignment
      ↓  summarise each cluster with an LLM
  6 level-1 summaries
      ↓  same again
  3 level-2 summaries
      ↓
  1 root
```

A chunk can belong to more than one cluster (the paper allows this: membership
probability ≥ 0.10). Answering uses the paper's **collapsed tree** approach —
every node at every level is a candidate, ranked by cosine similarity — so the
retriever chooses the level, rather than you choosing it in advance.

### Where this differs from the paper

| Paper | Here | Why |
|---|---|---|
| UMAP for dimension reduction | PCA | Ships with scikit-learn, behaves predictably on a few hundred nodes, one less dependency |
| Local + global clustering | Single-pass GMM per level | The corpus is hundreds of chunks, not thousands |
| GPT-3.5 summaries | Pluggable: Ollama / Claude / OpenAI / extractive | Runs with no key and no cost |
| QA over QuALITY, NarrativeQA | QA over your own PDFs | The point is a study tool |

## Quickstart

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

# put your PDFs here
copy "my-notes.pdf" data\raw\

python scripts/build_index.py       # chunk, embed, cluster, summarise
cd backend
uvicorn app.api:app --reload --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                         # http://localhost:5173
```

### 3. Better summaries (optional but recommended)

The default `offline` provider writes *extractive* summaries — it picks
representative sentences rather than writing new ones. That keeps the project
runnable with zero setup, but RAPTOR's whole premise is **abstractive**
summaries, so quality improves a lot with a real model:

```bash
# install Ollama from https://ollama.com, then
ollama pull llama3.2:3b

copy .env.example .env              # set LLM_PROVIDER=ollama
python scripts/build_index.py       # rebuild so the summaries are rewritten
```

Or set `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`, or
`LLM_PROVIDER=openai` with `OPENAI_API_KEY`.

## Asking from the terminal

```bash
python scripts/ask.py "what does Shift+F5 do in Aspen Plus?"
python scripts/ask.py --mode flat "what is the full-load slip?"
python scripts/ask.py --compare "what topics does the quiz cover overall?"
```

## Results

![Evaluation](docs/eval.png)

Run it yourself with `python eval/run_eval.py`; current numbers live in
[`eval/results.md`](eval/results.md) and are served to the web app from
`eval/results.json`.

The evaluation set is 26 hand-written questions over the demo corpus:
14 **specific** (one chunk holds the answer), 8 **broad** (needs several parts
of the corpus), 4 **unanswerable** (the corpus genuinely cannot answer them, so
the right response is to refuse).

Summaries written by `llama3.2:3b` through Ollama, answers by the same model:

| Run | Specific hit@6 | Specific MRR | Broad coverage | Refusal rate | Summary nodes / query |
|---|---|---|---|---|---|
| `flat` (baseline) | 1.000 | 0.738 | 0.938 | 1.000 | 0.00 |
| `tree` (RAPTOR) | 1.000 | 0.726 | 0.938 | 1.000 | 0.96 |
| `tree_hybrid` (+ BM25) | 0.929 | 0.786 | 0.875 | 1.000 | 0.92 |

Earlier run, with extractive summaries instead (`LLM_PROVIDER=offline`), for comparison:

| Run | Specific hit@6 | Specific MRR | Broad coverage | Refusal rate | Summary nodes / query |
|---|---|---|---|---|---|
| `flat` | 1.000 | 0.738 | 0.938 | 0.750 | 0.00 |
| `tree` | 1.000 | 0.726 | 0.938 | 0.500 | 0.77 |
| `tree_hybrid` | 0.929 | 0.786 | 0.812 | 0.250 | 0.50 |

**What changed when real summaries replaced extractive ones:** refusal rate went
from 0.25–0.75 to **1.000 across all three runs** — every unanswerable question
is now correctly declined instead of answered from irrelevant passages. The tree
is also genuinely used more (0.96 summary nodes per query, up from 0.77), and
every broad question except one retrieves at least one summary.

**What did not change:** the retrieval metrics are identical, because the leaf
chunks and their embeddings never changed — only the summaries did.

### What didn't work, and why

**The tree still does not beat the baseline on retrieval quality** — but the
reason is now clear, and it is the measurement, not the method.

1. **The broad metric is saturated.** Both `flat` and `tree` score 0.938, and 7
   of the 8 broad questions score a perfect 1.00 for both. A metric with no
   headroom cannot show a difference. The evidence quotes are keyword-ish
   (`"DSTWU"`, `"NRTL"`), and six chunks happen to contain them, so the flat
   baseline satisfies them too. Broad questions need required evidence spread
   across *more* of the corpus than six chunks can physically hold, which is the
   only situation where a 91-chunk summary has an advantage it cannot fake.
   **This is the next thing to fix, and it must be fixed before the comparison
   means anything.**
2. **Summary quality was a red herring for retrieval, but decisive for answers.**
   Replacing extractive summaries with written ones changed the retrieval
   numbers not at all, and the refusal rate from 0.25–0.75 to 1.000. The lesson:
   the summaries affect what the model *does with* retrieved context far more
   than what gets retrieved.
3. **The corpus is small and repetitive** — 192 chunks, and four of the five PDFs
   are variants of the same quiz. Near-duplicate filtering (cosine ≥ 0.95) was
   added because the top 6 results were otherwise four copies of one page; it
   skips 2 nodes on a typical query.

Also observed: 7 of the 14 *specific* questions retrieve a summary node, twice at
rank 1, which is exactly the failure mode you would fear — a broad summary
outranking the chunk holding the actual number. It costs a little MRR (0.726 vs
0.738) but never a hit, so the effect is real but small. A level-aware score
penalty is the obvious thing to try.

**Two summaries are still extractive.** `S1-001` and `S1-005` fell back when
their Ollama calls returned nothing, and the original code swallowed that
silently inside a 57-minute build. Failures are now reported as they happen and
counted in `meta.json`; those two nodes still need repairing.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Provider readiness, index stats, embedding dimension |
| `GET /api/tree` | Every node, trimmed for drawing |
| `GET /api/node/{id}` | One node plus the nodes it was summarised from |
| `POST /api/ask` | Answer as JSON (`mode`: `tree` or `flat`) |
| `POST /api/ask/stream` | The same, streamed as server-sent events |
| `POST /api/compare` | Both retrievers on one question |
| `GET /api/eval` | Latest evaluation results |
| `POST /api/upload` | Add a PDF (rebuild the index afterwards) |

## Project structure

```
backend/app/
  config.py     every setting, all with working defaults
  ingest.py     PDF -> cleaned text -> recursive chunks (leaf nodes)
  tree.py       cluster + summarise + recurse = the RAPTOR tree
  llm.py        one interface over offline / ollama / anthropic / openai
  retrieve.py   flat vs collapsed-tree search, BM25 hybrid, RRF, dedupe
  generate.py   citation-forced prompt, citation checking, streaming
  store.py      save and load nodes.json + embeddings.npy
  api.py        FastAPI endpoints
scripts/
  build_index.py, ask.py
eval/
  golden.jsonl, run_eval.py, results.json, results.md
frontend/src/
  views/        Ask, Tree, Compare, Evaluation
  components/   TreeMap (SVG), SourceCard, TracePanel, AnswerText
```

The frontend has **no UI or charting dependencies** — the tree is hand-drawn SVG,
so the whole bundle is React plus about 700 lines of application code.

## Notes

- `data/raw/` and `data/index/` are gitignored: course material stays off GitHub.
- The URL hash selects a view and can carry a question: `#tree`, `#ask?q=what%20is%20slip`.
- Vite proxies `/api` to port 8000 in development, so there is no CORS setup.

## Next steps

- [x] Rebuild with Ollama summaries and re-run the evaluation — done; refusal rate reached 1.000, retrieval unchanged
- [ ] Redesign the broad questions so the metric can discriminate (**the blocker**)
- [ ] Repair the two summaries that fell back to extractive text
- [ ] Try a level-aware score penalty so summaries stop outranking exact chunks
- [ ] Add a cross-encoder reranker over the top 50 candidates
- [ ] Index-time deduplication, so the four quiz variants collapse into one
- [ ] Dockerfile and a deployed demo link
