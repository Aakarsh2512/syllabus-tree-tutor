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
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "eval"

NODES_PATH = INDEX_DIR / "nodes.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.json"

# --- embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- language model ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
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
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", "10"))
# How much cluster text we hand to the summarizer at once.
SUMMARY_INPUT_CHARS = int(os.getenv("SUMMARY_INPUT_CHARS", "6000"))

# --- retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
# Two passages this similar are treated as the same passage, and only the
# better-ranked one is kept. Near-duplicate documents otherwise fill every slot.
DEDUPE_THRESHOLD = float(os.getenv("DEDUPE_THRESHOLD", "0.95"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
