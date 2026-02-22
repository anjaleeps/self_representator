---
title: Self Representator
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: true
---

Self Representator
==================

An ingestion + retrieval pipeline that builds a personal knowledge base from three sources—GitHub, Medium, and local documents—and serves answers via a Gradio chat UI. Data is stored in a persistent Chroma collection; embeddings and LLM calls use OpenAI-compatible endpoints.

What it does
------------
- Pulls profile + repo metadata, topics, languages, and README content from GitHub.
- Fetches Medium articles (metadata + markdown) via RapidAPI.
- Reads local docs (Markdown, PDF, txt) with per-type parsers and chunks them into nodes.
- Indexes everything into Chroma with a chosen embedding model.
- Retrieves top-k nodes and generates answers with a chat system prompt that “acts as” the user; optionally refines queries before retrieval.

Key code
--------
- `self_representator.py` – Orchestrates ingestion (`IngestionPipeline`), indexing (`Indexer`), and querying (`QueryEngine`).
- `data_loaders.py` – Source-specific loaders/parsers for GitHub, Medium, and local files.
- `models.py` – Pydantic models for typed API responses.
- `app.py` – Gradio entrypoint; starts ingestion and launches the chat UI.

Environment
-----------
Set in `.env` (example values):
- Identity & models: `USER_FULL_NAME`, `EMBED_MODEL`, `LLM_MODEL`, `OPENAI_API_KEY`, `GROQ_BASE_URL`, `GROQ_API_KEY`
- Vector store: `VECTOR_DB_NAME` (Chroma collection name)
- GitHub: `GITHUB_API_URL`, `GITHUB_API_TOKEN`, `GITHUB_USERNAME`
- Medium/RapidAPI: `MEDIUM_API_URL`, `MEDIUM_USER_ID`, `RAPID_API_KEY`
- Local docs: `LOCAL_DOC_FOLDER_PATH`

Run it
------
Start the Gradio chat (ingestion runs on startup):
```bash
python app.py
```
This launches a browser UI titled “<USER_FULL_NAME> Bot” that answers questions using the indexed context.

Notes
-----
- Chroma is persisted to the `chromadb/` directory by default.
- Query refinement happens before retrieval; retrieval logs the scored context chunks.
- Update the knowledge base anytime by rerunning `python app.py` (fresh ingestion) or calling `SelfRepresentator.update_knowledge()` in code.
