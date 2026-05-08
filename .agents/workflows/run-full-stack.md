---
description: how to run the full stack (backend + frontend) locally
---

## Prerequisites
- Python 3.11.x, Node.js 18+, and npm installed
- `.env` file at `d:\projects\bot\.env` fully configured (see `.env` for all required keys)
- GFR 2017 PDFs placed in `d:\projects\bot\data\`

## Steps

### Terminal 1 — Python FastAPI Backend

1. ```
   cd d:\projects\bot
   ```
2. ```
   venv\Scripts\activate
   ```
3. (First time only — ingest documents into ChromaDB):
   ```
   python backend\ingest.py
   ```
4. Start the API server on port 8080:
   ```
   uvicorn backend.main_api:app --host 0.0.0.0 --port 8080 --reload
   ```

### Terminal 2 — React Vite Frontend

1. ```
   cd d:\projects\bot\frontend
   ```
2. (First time only):
   ```
   npm install
   ```
3. Start the frontend dev server on port 5173:
   ```
   npm run dev
   ```

## Verify Everything is Working

| URL | Expected |
|---|---|
| http://localhost:8080 | `{"message":"CBRI ProcureBuddy API is running."}` |
| http://localhost:8080/api/health | `{"ok":true,...}` |
| http://localhost:5173 | ProcureBuddy Home page |
| http://localhost:5173/login | Login / Register page |
| http://localhost:5173/chat | Chatbot interface (login required) |

## Default Admin Access
Set `ADMIN_EMAIL` in `.env` to your email. Admin panel is accessible from the chatbot UI after login with that email.

## Architecture Flow
```
User Browser (port 5173)
      │
      ▼ /api/* (proxied by Vite)
Python FastAPI (port 8080)
      │
      ├─ SQLite DB (backend/chatbot.db)         ← Chat history & user auth
      ├─ ChromaDB (chroma_db/)                  ← RAG vector store
      └─ Groq API (LLaMA 3.1 via GROQ_API_KEY) ← LLM responses
```
