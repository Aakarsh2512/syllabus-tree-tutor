---
title: Syllabus Tree Tutor
emoji: 🌳
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: RAPTOR tree retrieval vs flat chunks, on any PDF
---

# Syllabus Tree Tutor

Upload a PDF, ask a question, and watch two retrievers answer it side by side:
ordinary chunk search, and **RAPTOR** tree-organized retrieval
([arXiv 2401.18059](https://arxiv.org/abs/2401.18059)), which clusters the chunks,
summarises each cluster, and searches every level of the resulting tree at once.

The demo loads two study guides (on RAG and on transformers). Use **Upload a PDF**
to try your own.

**This Space has no language model**, so answers are extractive — the most
relevant sentences rather than written prose. The retrieval comparison, which is
the point of the project, runs at full quality. Model-written answers,
the evaluation, and everything that did and didn't work are in the
[GitHub repository](https://github.com/Aakarsh2512/syllabus-tree-tutor).

Uploads are not private and are cleared when the Space restarts.
