---
description: how to ingest or re-ingest documents into the RAG knowledge base (ChromaDB)
---

## Prerequisites
- Python 3.11.x installed with `venv` activated
- PDF or supported documents placed inside `d:\projects\bot\data\`
- Supported formats: `.pdf`, `.txt`, `.md`, `.docx` (see `backend/ingest.py` for `SUPPORTED_DOC_EXTENSIONS`)

## Steps

1. Activate the virtual environment:
   ```
   cd d:\projects\bot
   venv\Scripts\activate
   ```

2. Place your new documents in the `data/` folder.

3. Run the ingestion script:
   ```
   python backend\ingest.py
   ```

4. The script will:
   - Load all supported documents from `data/`
   - Chunk the text into overlapping segments
   - Create embeddings using HuggingFace Sentence Transformers (runs **locally**, no API key needed)
   - Store the vectors persistently in `chroma_db/`

## Via Admin Panel (While Server is Running)

You can also upload documents and trigger re-indexing through the web UI:
1. Log in with the **admin email** (set in `.env` as `ADMIN_EMAIL`)
2. Navigate to **Admin Panel** from the sidebar
3. Use **Upload Documents** to add new files — this auto-triggers re-indexing
4. Use **Reindex** to rebuild the vector store from existing files in `data/`

## Notes
- Re-running `ingest.py` is safe — it rebuilds ChromaDB from scratch using current files in `data/`.
- Do NOT upgrade `chromadb` to >= 0.5 — causes SQLite schema conflicts.
- Embedding model: `sentence-transformers` (local, no internet required for inference).
