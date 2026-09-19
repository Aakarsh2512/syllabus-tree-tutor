# Syllabus Tree Tutor as one container: FastAPI serving both the API and the
# built React app on port 7860 (the port Hugging Face Spaces expects).
#
#   docker build -t syllabus-tree-tutor .
#   docker run -p 7860:7860 syllabus-tree-tutor
#
# Answers are extractive by default (LLM_PROVIDER=offline): free hosts cannot
# run a local model at a usable speed. Set LLM_PROVIDER=anthropic plus
# ANTHROPIC_API_KEY (or openai / OPENAI_API_KEY) as secrets for written answers.

# ---------------------------------------------------------------- frontend --
FROM node:22-slim AS frontend
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------- app ----
FROM python:3.12-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/app/data \
    LLM_PROVIDER=offline \
    DEMO_CORPUS_NAME="Demo: RAG and transformers study guides" \
    MAX_UPLOAD_BYTES=15728640 \
    MAX_UPLOAD_FILES=3

WORKDIR /home/user/app

# CPU-only PyTorch first. The default wheel on Linux bundles CUDA and is
# several gigabytes larger, for a GPU this container will never have.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Fetch the embedding model now, so the running app never needs the network.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user backend ./backend
COPY --chown=user scripts ./scripts
COPY --chown=user eval ./eval
COPY --chown=user deploy/demo ./data/raw
COPY --chown=user --from=frontend /frontend/dist ./frontend/dist

# Build the demo tree at image build time, so the first visitor never waits.
RUN python scripts/build_index.py

EXPOSE 7860
WORKDIR /home/user/app/backend
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
