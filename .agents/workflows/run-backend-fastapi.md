---
description: how to run the Python FastAPI backend (legacy / RAG engine)
---

## Prerequisites
- Python 3.11.x installed
- `.env` file at `d:\projects\bot\.env` with valid values for `GROQ_API_KEY`, `SMTP_*`, `ADMIN_EMAIL`, `ALLOWED_DOMAINS`
- GFR 2017 PDF documents placed inside `d:\projects\bot\data\`

## Steps

1. Open a terminal and navigate to the project root:
   ```
   cd d:\projects\bot
   ```

2. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

3. (First time only) Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (First time only, or when new PDFs are added) Ingest documents into ChromaDB:
   ```
   python backend\ingest.py
   ```
   This will load PDFs from `data/`, chunk them, create embeddings, and persist them in `chroma_db/`.

5. Start the FastAPI backend server (runs on port 8080 by default):
   ```
   uvicorn backend.main_api:app --host 0.0.0.0 --port 8080 --reload
   ```

6. Verify the server is running:
   - Open http://localhost:8080 → should return `{"message": "CBRI ProcureBuddy API is running."}`
   - Open http://localhost:8080/api/health → should return `{"ok": true, ...}`

## Notes
- The FastAPI backend is the **primary active backend** used by the React frontend.
- Do NOT upgrade `langchain` to >=1.0 or `numpy` to >=2.0 — this will break the RAG pipeline.
- Chat history is stored in `backend/chatbot.db` (SQLite via SQLAlchemy).
- Vector store is at `chroma_db/` (ChromaDB 0.4.x).
