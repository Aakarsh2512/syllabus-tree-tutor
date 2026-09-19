"""Every setting the project uses, in one place.

Values come from the environment (see .env.example), and every one has a
default that works, so the project runs without any configuration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# backend/app/config.py -> parents[2] is the project root
ROOT = Path(__file__).resolve().parents[2]
# Everything the app writes lives under DATA_DIR. A deployment, or a local
# test that must not touch the real index, points it somewhere else.
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = ROOT / "eval"
# One folder per uploaded corpus, so uploads never overwrite each other
# or the demo index.
CORPORA_DIR = DATA_DIR / "corpora"
DEMO_CORPUS_ID = "demo"
DEMO_CORPUS_NAME = os.getenv("DEMO_CORPUS_NAME", "Demo: course material")
# Where the built React app is, when FastAPI serves it itself.
FRONTEND_DIST = ROOT / "frontend" / "dist"

NODES_PATH = INDEX_DIR / "nodes.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.json"

# --- embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- language model ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Summarising a 46-chunk cluster on a CPU took longer than the old 300s limit,
# which is what silently broke two summaries in the first build.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- chunking (leaf nodes) ---
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]

# --- tree building ---
MAX_TREE_LEVELS = int(os.getenv("MAX_TREE_LEVELS", "4"))
# A node joins every cluster whose membership probability is at least this.
# The RAPTOR paper allows a chunk to belong to more than one cluster.
CLUSTER_PROB_THRESHOLD = float(os.getenv("CLUSTER_PROB_THRESHOLD", "0.10"))
MAX_CLUSTERS_PER_LEVEL = int(os.getenv("MAX_CLUSTERS_PER_LEVEL", "10"))
# Roughly how many nodes each summary should cover. 0 means "let BIC decide",
# which is what the paper does; a number forces finer or coarser summaries.
CLUSTER_TARGET_SIZE = int(os.getenv("CLUSTER_TARGET_SIZE", "0"))
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", "10"))
# How much cluster text we hand to the summarizer at once.
SUMMARY_INPUT_CHARS = int(os.getenv("SUMMARY_INPUT_CHARS", "6000"))

# --- retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
# Two passages this similar are treated as the same passage, and only the
# better-ranked one is kept. Near-duplicate documents otherwise fill every slot.
DEDUPE_THRESHOLD = float(os.getenv("DEDUPE_THRESHOLD", "0.95"))
# Subtracted from a summary node's cosine score before ranking. Summaries are
# broad, so they can outrank the one chunk that holds an exact number; a small
# penalty lets chunks win ties on narrow questions while summaries still win
# clearly on broad ones. 0.0 disables it.
SUMMARY_PENALTY = float(os.getenv("SUMMARY_PENALTY", "0.0"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))
# Shorter answers matter on a CPU model: generation is the slow part.
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "500"))
# With no language model, refuse when the best retrieved passage scores below
# this. It is a weak signal: measured on the demo, an unanswerable question
# ("attendance policy", 0.337) outscored an answerable one ("what does this
# cover", 0.245), so no threshold separates them cleanly. 0.20 keeps every
# answerable question tested and refuses most, but not all, unanswerable ones.
OFFLINE_MIN_SCORE = float(os.getenv("OFFLINE_MIN_SCORE", "0.20"))


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(40 * 1024 * 1024)))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
