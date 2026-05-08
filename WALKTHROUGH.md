# 🏗️ ProcureBuddy — Complete Project Walkthrough

> **ProcureBuddy** is an AI-powered procurement chatbot built for CSIR (Council of Scientific and Industrial Research). It answers procurement policy questions using a RAG (Retrieval-Augmented Generation) pipeline grounded in official GFR 2025 and CSIR Manual 2019 documents.

---

## 🔗 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER BROWSER (:5173)                         │
│              React 18 + Vite + Tailwind CSS                      │
└────────────────┬───────────────────────────────┬─────────────────┘
                 │ /api/* (Vite proxy)           │ /ai/* (Vite proxy)
                 ▼                               ▼
┌────────────────────────────┐    ┌──────────────────────────────┐
│  SPRING BOOT BACKEND (:8080) │    │  PYTHON AI SERVICE (:8000)   │
│  Java 17, Spring Boot 3.4    │───▶│  FastAPI + uvicorn            │
│  JPA/Hibernate, PostgreSQL   │    │  FAISS + Sentence-Transformers│
│  REST API, Auth, Chat Mgmt   │    │  Groq LLM (Llama3-8B)        │
└──────────────┬───────────────┘    └───────────────┬──────────────┘
               │                                    │
               ▼                                    ▼
    ┌──────────────────┐              ┌─────────────────────────┐
    │  PostgreSQL RDS   │              │  data/ (PDF Knowledge   │
    │  (AWS ap-south-1) │              │  Base) + FAISS Index    │
    └──────────────────┘              └─────────────────────────┘
```

| Layer | Tech Stack | Port | Purpose |
|-------|-----------|------|---------|
| **Frontend** | React 18, Vite 5, Tailwind CSS, Framer Motion | `:5173` | Chat UI, Auth, Admin dashboard |
| **Backend** | Spring Boot 3.4, JPA/Hibernate, PostgreSQL | `:8080` | REST API, User/Chat/Feedback management |
| **AI Service** | Python 3.11, FastAPI, FAISS, Sentence-Transformers, Groq | `:8000` | RAG retrieval, LLM answer generation |

---

## 🔄 Request Flow — How a Chat Message Works

```
User types "What is the LTE threshold?"
        │
        ▼
1. Frontend (App.jsx) ─── POST /api/chats/{id}/message ───▶ Vite Proxy
        │
        ▼
2. Spring Boot (ChatController.java) ─── receives request
        │
        ▼
3. ChatService.sendMessageAsync() ─── dispatched to @Async thread pool
        │
        ▼
4. PythonAiService.chat() ─── POST http://127.0.0.1:8000/chat
        │
        ▼
5. Python /chat endpoint ─── detect_intent() → "PROCESS"
        │
        ▼
6. question_transformer_node() ─── expand query + detect amount + GFR slab
        │
        ▼
7. multi_query_retrieval_node() ─── hybrid search (semantic + keyword + RRF)
        │
        ▼
8. rerank_node() ─── Flashrank / CrossEncoder / heuristic rerank
        │
        ▼
9. threshold_logic_node() ─── inject GFR 2025 threshold table
        │
        ▼
10. agentic_generation_node() ─── Groq API call (Llama3-8B, timeout=90s)
        │
        ▼
11. post_process_llm_output() ─── parse 5-section format + anti-hallucination
        │
        ▼
12. ChatResponse returned to Spring ─── JSON with response + source_chunk_ids
        │
        ▼
13. Spring saves MessageEntity + MessageRevisionEntity to PostgreSQL
        │
        ▼
14. CompletableFuture resolves ─── response sent back to browser
```

---

## 📁 Complete File & Folder Reference

---

### 📂 Root Directory — `d:\projects\bot\`

| File/Folder | Purpose |
|-------------|---------|
| `.env` | 🔑 **Master environment variables** — Groq API key, DB credentials, SMTP config, JWT secret, CORS settings |
| `.gitignore` | Git ignore rules (venv, node_modules, __pycache__, .env, etc.) |
| `README.md` | Project documentation and setup instructions |
| `WALKTHROUGH.md` | **This file** — Complete project walkthrough |
| `Dockerfile` | 🚫 Legacy Docker build (for old Python backend) — **not actively used** |
| `requirements.txt` | 🚫 Legacy root-level Python dependencies — **not actively used** |
| `app.py` | 🚫 Legacy Streamlit app entry point — **deprecated** |
| `main.py` | 🚫 Legacy Groq test script — **deprecated** |
| `ui.py` | 🚫 Legacy Streamlit UI — **deprecated** |
| `chatbot.db` | 🚫 Legacy SQLite database — **replaced by PostgreSQL** |
| `chat_history.json` | 🚫 Legacy chat history file — **deprecated** |
| `conversations.json` | 🚫 Legacy conversations — **deprecated** |
| `users.json` | 🚫 Legacy user store — **deprecated** |
| `venv/` | Python virtual environment (all Python dependencies installed here) |
| `chroma_db/` | 🚫 Legacy ChromaDB vector store — **replaced by FAISS** |
| `__pycache__/` | Python bytecode cache |

---

### 📂 `frontend/` — React + Vite + Tailwind CSS Frontend

The user-facing single-page application with two main routes:
- `/` → Landing page (Home)
- `/chat` → Chat workspace (App)

#### Root Config Files

| File | Purpose |
|------|---------|
| `package.json` | **NPM config** — Dependencies: React 18, Framer Motion, react-markdown, react-router-dom v6, remark-gfm, uuid, Axios |
| `vite.config.js` | **Vite dev server** — Port 5173, proxies `/api` → `:8080` (Spring) and `/ai` → `:8000` (Python) |
| `tailwind.config.js` | **Tailwind theme** — Custom design tokens, colors, spacing, fonts |
| `postcss.config.js` | PostCSS pipeline (Tailwind + Autoprefixer) |
| `index.html` | HTML shell — mounts React app at `<div id="root">` |
| `eslint.config.js` | ESLint rules for code quality |

#### `src/` — Application Source Code

| File | Lines | Core Work |
|------|-------|-----------|
| `main.jsx` | 39 | **React entry point** — `ReactDOM.createRoot`, sets up `BrowserRouter` with 2 routes (`/` → Home, `/chat` → App), wraps everything in `ThemeProvider` + `SeasonalProvider` |
| `index.css` | ~800 | **Global stylesheet** (25KB) — CSS custom properties for light/dark themes, Tailwind base layer, all custom utility classes |
| `App.jsx` | 365 | **Main chat workspace** — Session management (localStorage), chat list CRUD, message send/receive, PDF export, thumbs-up/down feedback, draft chats, auto-refresh |
| `Home.jsx` | ~500 | **Landing page** — Hero section with physics-based particle animations, feature cards, CSIR branding, CTA button to `/chat` |
| `LoginPage.jsx` | ~700 | **Authentication UI** — Login form, Register (with OTP email verification), Reset Password, TOTP 2FA setup with QR code |
| `Views.jsx` | ~1800 | **All in-app views** — `ChatView` (message list, markdown rendering with react-markdown, typing indicator, feedback buttons), `SettingsView` (theme/seasonal mode), `ProfileModal` (avatar, display name), `AdminView` (document upload, reindex) |
| `Sidebar.jsx` | ~500 | **Sidebar navigation** — Chat list with search, folders, new chat button, pin/delete/move actions, collapsible toggle |
| `api.js` | 123 | **API client** — 22 REST endpoints with JWT token auto-injection from localStorage, fetch wrapper with error handling |
| `physics.js` | ~150 | **Physics engine** — Spring-based particle animation for the landing page hero section |
| `SeasonalLayer.jsx` | ~100 | **Seasonal overlay** — Festival-aware particle effects (Holi colors, Diwali lights, etc.) |
| `HumanVerificationSlider.jsx` | ~200 | **CAPTCHA slider** — Drag-to-verify component on login page |
| `JellySlider.jsx` | ~170 | **Animated slider** — Jelly-physics range slider widget |
| `JellySwitch.jsx` | ~130 | **Animated toggle** — Jelly-physics on/off switch |

#### `src/components/` — Shared UI Components

| File | Core Work |
|------|-----------|
| `Layout.jsx` | **App shell** — Sidebar + Topbar + Main content area, responsive breakpoints, mobile drawer |
| `Topbar.jsx` | **Top navigation bar** — Shows chat title, settings gear, profile avatar, logout button |
| `ui.jsx` | **Micro-components** — Reusable Button, Input, Card, Badge with consistent styling |

#### `src/context/` — React Context Providers

| File | Core Work |
|------|-----------|
| `ThemeContext.jsx` | **Theme system** — Light / Dark / System auto-detect mode, switches CSS variables on `<html>`, persists choice in localStorage |
| `SeasonalContext.jsx` | **Seasonal/festival system** (35KB) — Indian festival calendar (Holi, Diwali, Republic Day, Independence Day, etc.), auto-detects current festival, applies themed particle effects and color schemes |

#### `src/config/`

| File | Core Work |
|------|-----------|
| `api.js` | **API base URL** — Empty string (same-origin), relies on Vite proxy for routing |

---

### 📂 `backend-spring/` — Spring Boot 3.4 Backend (Java 17)

The primary backend handling all business logic, user management, database persistence, and Python AI service bridge.

#### Root Files

| File | Purpose |
|------|---------|
| `pom.xml` | **Maven config** — Spring Boot 3.4.4, spring-boot-starter-web, spring-boot-starter-data-jpa, spring-boot-starter-validation, spring-boot-starter-mail, PostgreSQL driver, PDFBox 3.0.3, Flyway, Lombok, ZXing (QR codes for TOTP), Azure WebApp deploy plugin |
| `run-local.ps1` | PowerShell script to launch Spring Boot locally with env vars |
| `settings.xml` | Maven repository settings |
| `application.yml` | **All Spring config** — PostgreSQL RDS connection, CORS origins, async thread pool, Python service URL + timeouts (read=180s, connect=10s), Hikari connection pool, SMTP mail |

#### `config/` — Configuration Beans (4 files)

| File | Core Work |
|------|-----------|
| `ProcureBuddyApplication.java` | `@SpringBootApplication` — Application entry point, `main()` method |
| `ProcureBuddyProperties.java` | `@ConfigurationProperties(prefix="procurebuddy")` — Type-safe config: CORS origins, Python service base URL + timeouts (connect=10s, read=180s), async pool sizes (core=16, max=32, queue=400), admin email |
| `AsyncCacheConfig.java` | `@EnableAsync` + `@EnableCaching` — Creates `aiTaskExecutor` (ThreadPoolTaskExecutor for async chat processing), `ConcurrentMapCacheManager` for chat list/message caching |
| `CorsConfig.java` | CORS filter — allows configured origins with all methods/headers |
| `AppBeansConfig.java` | Additional Spring beans (ObjectMapper configuration) |

#### `controller/` — REST API Endpoints (9 files)

| File | Base Path | Endpoints | Core Work |
|------|-----------|-----------|-----------|
| `AuthController.java` | `/api/auth` | POST `/register/start`, `/register/verify`, `/login`, `/change-password`, `/reset-password`, `/profile`, `/totp/setup`, `/totp/enable`, `/totp/verify`, `/totp/disable`; GET `/status` | Full auth lifecycle: OTP-based registration, login, password management, TOTP 2FA |
| `ChatController.java` | `/api/chats` | GET `/`, `/{chatId}`; POST `/{chatId}/message`, `/{chatId}/regenerate`, `/pin`; DELETE `/{chatId}`; GET `/{chatId}/export` | Chat CRUD, async message sending, response regeneration, PDF export |
| `FeedbackController.java` | `/api/feedback` | POST `/` | Submit thumbs-up/thumbs-down on AI responses |
| `AdminController.java` | `/api/admin` | GET `/documents`, `/status`; POST `/upload`, `/reindex` | Admin-only: list/upload documents, trigger knowledge base reindex |
| `DocumentController.java` | `/api/documents` | POST `/search` | Search the knowledge base (proxied to Python) |
| `FolderController.java` | `/api/folders` | GET `/`; POST `/`, `/{folderId}/move`; DELETE `/{folderId}` | Create/delete/list folders, move chats between folders |
| `MemoryController.java` | `/api/memory` | GET `/`; POST `/`; DELETE `/{id}` | User preference/memory storage CRUD |
| `AnalyticsController.java` | `/api/analytics` | GET `/prompts` | Prompt frequency analytics for dashboard |
| `HealthController.java` | `/api/health` | GET `/` | Health check endpoint |

#### `service/` — Business Logic Layer (11 files)

| File | Core Work |
|------|-----------|
| `ChatService.java` | **Core chat logic** — `sendMessageAsync()` dispatches to `@Async("aiTaskExecutor")`, calls `PythonAiService.chat()`, saves `MessageEntity` + `MessageRevisionEntity`, auto-generates titles from first message, pagination support |
| `PythonAiService.java` | **Python AI bridge** — Java `HttpClient` to Python FastAPI at `:8000`. Endpoints: `/chat`, `/search`, `/reload`, `/health`. Features: retry across `localhost`↔`127.0.0.1`, error code mapping (429→TOO_MANY_REQUESTS, 504→GATEWAY_TIMEOUT, 503→SERVICE_UNAVAILABLE), timeout→GATEWAY_TIMEOUT distinction |
| `AuthService.java` | **Authentication** — OTP email verification (6-digit code, 10min expiry), BCrypt password hashing, TOTP 2FA via Google Authenticator (QR code generation with ZXing), domain whitelist validation, admin detection |
| `FeedbackService.java` | **Feedback loop** — Stores thumbs-up/down per message, builds `FeedbackAwareChatContext` with blocked chunk IDs + response hashes so regenerated answers avoid previously disliked responses |
| `AdminService.java` | **Admin operations** — Verifies admin email, uploads PDFs to `data/` folder, triggers Python `/reload` to rebuild FAISS index, lists all indexed documents |
| `DocumentService.java` | **Document search** — Proxies search queries to Python `/search`, maps `SearchMatch` results to `RetrievedChunk` |
| `ChatExportService.java` | **PDF export** — Generates formatted PDF from chat history using Apache PDFBox, includes user messages + AI responses with timestamps |
| `FolderService.java` | Chat folder CRUD — create, delete, list folders per user |
| `MemoryService.java` | User memory/preferences — store and retrieve user-specific notes |
| `OtpMailService.java` | **SMTP email** — Sends OTP verification codes via Gmail SMTP |
| `PromptAnalyticsService.java` | **Analytics** — Tracks prompt text frequency in `prompt_stats` table |

#### `entity/` — JPA Entities / Database Tables (12 files)

| File | DB Table | Columns | Core Work |
|------|----------|---------|-----------|
| `UserEntity.java` | `users` | id, email (unique), password_hash, display_name, username, avatar_base64, is_admin, totp_secret | User accounts with 2FA support |
| `ChatEntity.java` | `chats` | id (UUID), user_id (FK), title, preview, pinned, folder_id (FK), updated_at | Chat sessions belonging to users |
| `MessageEntity.java` | `messages` | id, chat_id (FK), message (user text), response (AI text), response_id, response_hash, source_chunk_ids, timestamp | User↔AI message pairs |
| `MessageRevisionEntity.java` | `message_revisions` | id, message_id (FK), response, source ("initial"/"regenerated") | Revision history for regenerated responses |
| `FeedbackEntity.java` | `feedback` | id, user_email, chat_id, message_id, type ("like"/"dislike") | Per-message user feedback |
| `FolderEntity.java` | `folders` | id, user_id (FK), name | Chat organization folders |
| `DocumentEntity.java` | `documents` | id, file_name, file_size, uploaded_at | Uploaded document metadata |
| `DocumentChunkEntity.java` | `document_chunks` | id, document_id (FK), content, chunk_index | Document text chunks |
| `KnowledgeChunkEntity.java` | `knowledge_chunks` | id, document_id, file_name, content, token_count, embedding | Knowledge base chunks with vector embeddings |
| `MemoryEntity.java` | `memories` | id, user_id (FK), key, value, updated_at | User-specific memory/preferences |
| `PendingOtpEntity.java` | `pending_otps` | id, email, otp_code, created_at | Temporary OTP codes (10min expiry) |
| `PromptStatEntity.java` | `prompt_stats` | id, prompt_text, count, last_used | Prompt frequency tracking |

#### `repository/` — Spring Data JPA Repositories (12 files)

| File | Core Work |
|------|-----------|
| `UserRepository.java` | `findByEmail()` — Primary user lookup |
| `ChatRepository.java` | `findAllByUserOrderByPinnedDescUpdatedAtDesc()` — Chat list sorted by pinned first, then recent |
| `MessageRepository.java` | `findAllByChatOrderByTimestampAscIdAsc()` — Messages in chronological order, `countByChat()`, batch `countAllByChatIds()` |
| `MessageRevisionRepository.java` | `deleteAllByMessageChat()` — Cascade delete revisions |
| `FeedbackRepository.java` | Feedback queries by user/chat/message, `deleteAllByChatId()` |
| `FolderRepository.java` | `findByIdAndUser()`, `findAllByUser()` — Folder CRUD |
| `DocumentRepository.java` | Document metadata queries |
| `DocumentChunkRepository.java` | Document chunk queries |
| `KnowledgeChunkRepository.java` | `findByDocumentId()` — Knowledge chunk queries |
| `MemoryRepository.java` | `findAllByUser()`, `findByUserAndKey()` — User memory queries |
| `PendingOtpRepository.java` | `findByEmail()` — OTP lookup for verification |
| `PromptStatRepository.java` | Prompt frequency aggregation queries |

#### `dto/request/` — Incoming Request Payloads (18 files)

| File | Used By |
|------|---------|
| `SendMessageRequest.java` | ChatController — `{user, message}` |
| `LoginRequest.java` | AuthController — `{email, password}` |
| `RegisterStartRequest.java` | AuthController — `{email}` |
| `RegisterVerifyRequest.java` | AuthController — `{email, otp, password}` |
| `ChangePasswordRequest.java` | AuthController — `{email, newPassword}` |
| `ResetPasswordRequest.java` | AuthController — `{email}` |
| `UpdateProfileRequest.java` | AuthController — `{email, displayName, username, avatarBase64}` |
| `FeedbackRequest.java` | FeedbackController — `{user, chatId, messageId, type}` |
| `DocumentSearchRequest.java` | DocumentController — `{query}` |
| `CreateFolderRequest.java` | FolderController — `{name, user}` |
| `MoveChatRequest.java` | FolderController — `{chatId, folderId, user}` |
| `PinChatRequest.java` | ChatController — `{chatId, pinned, user}` |
| `MemoryRequest.java` | MemoryController — `{user, key, value}` |
| `TotpSetupRequest.java` | AuthController — `{email}` |
| `TotpEnableRequest.java` | AuthController — `{email, secret, code}` |
| `TotpVerifyRequest.java` | AuthController — `{email, code}` |
| `TotpDisableRequest.java` | AuthController — `{email}` |
| `LegacyRegisterRequest.java` | AuthController — Combined register flow |

#### `dto/response/` — Outgoing Response Payloads (3 files)

| File | Core Work |
|------|-----------|
| `ChatMessageResponse.java` | `{id, role, content, timestamp}` — Individual message |
| `ChatSummaryResponse.java` | `{chatId, title, preview, messageCount, updatedAt, isPinned, folderId}` — Chat list items |
| `FolderResponse.java` | `{id, name, chatCount}` — Folder list items |

#### `exception/` — Error Handling (3 files)

| File | Core Work |
|------|-----------|
| `ApiException.java` | Custom runtime exception with `HttpStatus` code |
| `ErrorResponse.java` | Standardized JSON error envelope: `{status, message}` |
| `GlobalExceptionHandler.java` | `@ControllerAdvice` — Catches all exceptions, returns clean JSON errors to frontend |

#### `util/` — Utilities (2 files)

| File | Core Work |
|------|-----------|
| `UserResolver.java` | `requireByEmail()` / `requireByIdentifier()` — Resolves UserEntity, throws 404 if missing |
| `PasswordRules.java` | Password strength validation (min length, complexity rules) |

#### `persistence/` (1 file)

| File | Core Work |
|------|-----------|
| `FloatArrayStringConverter.java` | JPA AttributeConverter — Stores `float[]` embeddings as comma-separated strings in database columns |

#### `deploy/` (1 file)

| File | Core Work |
|------|-----------|
| `nginx-procurebuddy.conf.example` | Sample Nginx reverse proxy config for production deployment |

---

### 3.4 📂 `python-ai-service/` — FastAPI RAG Engine (Modularized v2.0)

The AI brain of ProcureBuddy. Runs as an independent microservice at `:8000`. The original 1,800-line monolith (`main_legacy.py`) has been refactored into a clean modular package.

#### Root Files

| File | Purpose |
|------|---------|
| `main.py` | **Backward-compatible entry point** — Thin wrapper that re-exports `app` from `app.main` so `uvicorn main:app` still works |
| `main_legacy.py` | 🚫 **Preserved monolith** (~76KB) — The original single-file RAG engine, kept for reference only |
| `requirements.txt` | Dependencies: `fastapi`, `uvicorn[standard]`, `langchain`, `langchain-community`, `langchain-groq`, `langgraph`, `sentence-transformers`, `flashrank`, `faiss-cpu`, `pypdf`, `pdfplumber`, `numpy`, `groq`, `python-dotenv`, `cachetools` |

#### `app/` — Modular Package Structure

```
python-ai-service/
├── main.py                           # uvicorn entry point (re-exports app.main.app)
├── main_legacy.py                    # preserved monolith (reference only)
├── requirements.txt
└── app/
    ├── __init__.py
    ├── main.py                       # FastAPI app factory + startup
    ├── api/
    │   ├── __init__.py
    │   └── chat_router.py            # /chat, /search, /reload, /health endpoints
    ├── core/
    │   ├── __init__.py
    │   ├── config.py                 # Environment config + Settings singleton
    │   ├── constants.py              # GFR 2025 thresholds, prompts, domain terms
    │   ├── rag_engine.py             # Expert RAG pipeline (scoring, reranking, generation)
    │   └── response_builder.py       # Response formatting, section parsing, cleanup
    ├── services/
    │   ├── __init__.py
    │   ├── knowledge_base.py         # FAISS index management, search, reload
    │   └── llm_service.py            # Groq LLM API calls with timeout/retry
    └── utils/
        ├── __init__.py
        ├── processors.py             # Intent detection, query expansion, tokenization
        └── text_cleaner.py           # Text cleaning, chunk quality audit, noise filtering
```

#### `app/main.py` — Application Entry Point

| Section | Core Work |
|---------|-----------|
| FastAPI app setup | Creates the app with title "ProcureBuddy AI Service", version 2.0.0 |
| Router registration | Includes `chat_router` at root level (no prefix) for backward compatibility |
| Startup event | Initializes KnowledgeBase from `data/`, logs config, validates API key |

#### `app/api/chat_router.py` — API Endpoints

| Endpoint | Method | Core Work |
|----------|--------|-----------|
| `/health` | GET | Returns knowledge base status (document count, chunk count, ready state) |
| `/reload` | POST | Rebuilds FAISS index from `data/` PDFs, clears answer cache |
| `/search` | POST | Hybrid retrieval with RRF fusion, returns ranked matches |
| `/chat` | POST | **Main entry** — greeting detection → agentic RAG pipeline → cached response |

Request/Response models:
- `ChatRequest` — `{message, user, displayName, username, email, bypass_cache, blocked_chunk_ids, blocked_response_hashes}`
- `ChatResponse` — `{response, response_id, response_hash, source_chunk_ids, metadata}`
- Answer cache: TTLCache (default 500 entries, 1h TTL)

#### `app/core/config.py` — Environment Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `data_dir` | Auto-discovered `../data/` | Knowledge base PDF directory |
| `index_dir` | `data/.index/` | Persisted FAISS index location |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding model |
| `chunk_size_words` | 350 | Words per chunk when splitting documents |
| `chunk_overlap_words` | 50 | Overlapping words between chunks |
| `top_k` | 5 | Number of chunks to return to LLM |
| `min_score` | 0.15 | Minimum cosine similarity to keep a match |
| `answer_cache_size` | 500 | Max cached answers |
| `answer_cache_ttl_seconds` | 3600 | Cache entry TTL (1 hour) |
| `llm_model` | `llama3-8b-8192` | Groq LLM model identifier |

#### `app/core/constants.py` — GFR 2025 Master Reference

| Section | Core Work |
|---------|-----------|
| `GFR_2025_TRUTH_TABLE` | 4-slab tuple: Direct Purchase (≤50K), LPC (50K–5L), LTE (5L–25L), OTE (>25L) — each with keywords, rule refs, value bands |
| `GFR_2025_REFERENCE_THRESHOLDS` | Legacy threshold conflict resolution map (25K, 2.5L, 5L, 25L, 50L) |
| `PROCUREMENT_THRESHOLDS_TABLE` | Markdown table injected into every LLM prompt as ground truth |
| `SYSTEM_PROMPT` | 5-section response format: Quick Answer, Rule Priority, Why This Applies, Process, Source Basis |
| `PROCESS_PROMPT` | Amount-based query prompt with threshold cross-checking |
| `WORKFLOW_PROMPT` | Step-by-step SOP prompt for procedural questions |
| `PROCUREMENT_DOMAIN_TERMS` | 30-term frozenset for domain gating |
| `SECTION_HEADING_RE` | Regex for procurement section heading detection |
| `GREETING_WORDS` | Frozenset of greeting tokens that bypass RAG |
| Scoring constants | `GFR_2025_RECENCY_BONUS=0.30`, `HEADING_BOOST=0.15`, `KEYWORD_OVERLAP_WEIGHT=0.07`, `DOMAIN_DENSITY_CAP=0.10`, `QUERY_RELEVANCE_CAP=0.20` |

#### `app/core/rag_engine.py` — Expert RAG Pipeline (~1,645 lines)

The core intelligence module implementing the full agentic RAG workflow:

```
/chat request
    │
    ├── question_transformer_node()
    │       • Expand query keywords based on intent
    │       • Extract amount (lakhs) → determine GFR slab
    │       • Generate retrieval query variations (LCEL or deterministic)
    │
    ├── multi_query_retrieval_node()
    │       • LangChain MultiQueryRetriever (if available)
    │       • OR deterministic multi-query with variations
    │       • Hybrid search: semantic (FAISS cosine) + keyword (BM25-style)
    │       • Reciprocal Rank Fusion (RRF) to merge rankings
    │
    ├── route_after_retrieval → retry_search_fallback_node() (if no matches)
    │       • Progressively relaxed search with lower min_score
    │
    ├── rerank_node()
    │       • Flashrank cross-encoder rerank (if available)
    │       • OR sentence-transformers CrossEncoder
    │       • OR local heuristic rerank (action-density, GFR recency, procurement verbs)
    │       • Semantic deduplication (SequenceMatcher ≥0.80 threshold)
    │
    ├── threshold_logic_node()
    │       • Inject PROCUREMENT_THRESHOLDS_TABLE into context
    │       • Attach GFR slab metadata to state
    │
    └── agentic_generation_node()
            • Select prompt template based on intent (SYSTEM / PROCESS / WORKFLOW / ANALYTICAL)
            • Build context from top-k matches with sentence selection
            • Groq API call with timeout=90s
            • Parse 5-section format
            • Anti-hallucination: fix "2.5 lakh" → "5 lakh" LTE
            • Fallback: rule-based answer if LLM fails
```

Key scoring functions:

| Function | Core Work |
|----------|-----------|
| `prioritize_matches()` | 7-signal scoring: keyword overlap, heading bonus, domain density, source priority, GFR 2025 recency, query relevance, text quality, analytical definition bonus |
| `filter_matches()` | Removes blocked chunks, noisy text, off-domain content, scientist name lists, table-of-contents pages |
| `rerank_matches()` | Action-sentence density scoring, GFR file boosting, procurement verb presence check, semantic dedup |
| `_rrf_fuse()` | Reciprocal Rank Fusion to combine semantic + keyword search rankings |
| `_ensure_analytical_coverage()` | For comparison queries, ensures both methods have ≥2 representative chunks |

#### `app/core/response_builder.py` — Response Formatting (~16KB)

| Function | Core Work |
|----------|-----------|
| `parse_llm_sections()` | Extract 5 named sections from LLM output |
| `render_structured_response()` | Format sections into final Markdown |
| `select_relevant_sentences()` | Choose the most relevant sentences from a chunk for context injection |
| `summarize_match_for_context()` | Compress a chunk into a context summary |
| `build_default_procedural_steps()` | Generate fallback procedural steps from knowledge base |
| `compact_answer_text()` | Clean up verbose LLM output |
| `extract_markdown_table_block()` | Extract and preserve Markdown tables from LLM output |

#### `app/services/knowledge_base.py` — FAISS Index Manager (~17KB)

| Method | Core Work |
|--------|-----------|
| `initialize()` | Load persisted FAISS index from `data/.index/` or trigger full rebuild |
| `reload()` | Full rebuild: extract text from all PDFs/TXT/DOCX in `data/`, chunk, embed, build FAISS index |
| `search_semantic()` | FAISS cosine similarity search against sentence-transformer embeddings |
| `search_keyword()` | BM25-style keyword matching with term-frequency scoring |
| `status()` | Returns document count, chunk count, ready state |
| `_persist_to_disk()` | Save FAISS index + chunk metadata JSON to `data/.index/` |
| `_load_from_disk()` | Restore FAISS index + chunk metadata from persisted files |

Index files:
- `data/.index/faiss.index` — Binary FAISS index
- `data/.index/chunks.json` — Chunk metadata (text, document_id, file_name, chunk_index, token_count)

#### `app/services/llm_service.py` — Groq LLM Client (~2.5KB)

| Function | Core Work |
|----------|-----------|
| `generate_llm_response()` | Groq API call with `timeout=90s`, handles `RateLimitError` → HTTP 429, `APITimeoutError` → HTTP 504 |

#### `app/utils/processors.py` — Intent & Query Processing (~11.5KB)

| Function | Core Work |
|----------|-----------|
| `detect_intent()` | Classifies query: `GREETING`, `PROCESS`, `ANALYTICAL`, `WORKFLOW`, `GENERAL` |
| `extract_amount_lakhs()` | Parses rupee amounts from text → float in lakhs |
| `expand_query_keywords()` | Intent-aware query expansion with procurement terms |
| `amount_to_context_keywords()` | Maps amount to GFR slab keywords |
| `extract_analytical_terms()` | Detects comparison targets (LTE vs OTE, LPC vs STE, etc.) |
| `get_analytical_method_variants()` | Returns all aliases for a procurement method |
| `tokenize()` | Text tokenizer for keyword overlap scoring |
| `extract_username()` | Resolves display name from user metadata |
| `semantic_dedup_key()` | Generates deduplication keys for semantic similarity |

#### `app/utils/text_cleaner.py` — Text Quality Utilities (~12.5KB)

| Function | Core Work |
|----------|-----------|
| `clean_text()` | Remove mojibake, normalize whitespace, fix encoding issues |
| `is_clean_chunk()` | Quality gate — rejects chunks with >50% digits, repeated characters, or no alphabetic content |
| `audit_chunk_quality()` | Returns quality metrics: action_sentence_density, numeric_reference_count, tag (ACTIONABLE/REFERENCE_ONLY), discard flag |
| `looks_like_table_of_contents()` | Detects TOC-style text (page numbers, dot leaders) |
| `has_definition_style()` | Detects definition-style text useful for comparison queries |
| `legalistic_noise_penalty()` | Penalizes chunks with excessive legal jargon |
| `contains_scientist_list()` | Detects name-list chunks (committee member lists) |

---

### 3.5 📂 `data/` — Knowledge Base Documents

The PDF corpus that feeds the FAISS vector index. These are the **ground truth** documents.

| Document | Size | Content |
|----------|------|---------|
| `UpdatedGFR31July2025_0.pdf` | 2.5 MB | **General Financial Rules 2025** — Master procurement thresholds (highest authority) |
| `CSIR Manual on procurement of goods 2019.pdf` | 6.6 MB | **CSIR Procurement Manual** — Institution-specific procurement rules |
| `amendments.pdf` | 6.1 MB | GFR amendments and office memorandums |
| `Compendium-Stores-&-Inventory-DSC.pdf` | 9.2 MB | Stores & inventory management compendium |
| `SnT special provisions.pdf` | 760 KB | S&T special procurement provisions (Part 1) |
| `SnT special provisions 2.pdf` | 326 KB | S&T special procurement provisions (Part 2) |
| `SnT special provisions 3.pdf` | 4.5 MB | S&T special procurement provisions (Part 3) |
| `Make in India preference.pdf` | 1.1 MB | Make in India / Local supplier preference policies |
| `Write-off.pdf` | 1.2 MB | Write-off procedures for obsolete/unserviceable assets |
| `.index/` | — | Persisted FAISS index (`faiss.index`) + chunk metadata (`chunks.json`) |

**Source priority when documents conflict**: GFR 2025 Truth Table > CSIR Manual 2019 > Make in India / SnT Special Provisions

---

### 3.6 📂 `backend/` — Legacy Python/FastAPI Backend (⚠️ DEPRECATED)

> ⚠️ This is the **old monolithic backend**, replaced by `backend-spring/` (Java) + `python-ai-service/` (Python). Kept for reference only.

| File | Core Work |
|------|-----------|
| `main_api.py` | Old FastAPI entry point with all endpoints (auth, chat, admin) in one file |
| `core.py` | Old RAG engine (~55KB) — ChromaDB-based retrieval, Groq LLM integration |
| `auth.py` | Old JWT-based authentication with SQLite |
| `config.py` | Old configuration loader |
| `database.py` | Old SQLAlchemy setup with SQLite |
| `models.py` | Old SQLAlchemy ORM models (User, Chat, Message) |
| `ingest.py` | Old document ingestion into ChromaDB |
| `oq.py` | Groq API connectivity test |
| `Dockerfile` | Docker build for old backend |
| `.env` | Old environment variables |
| `chroma_db/` | Old ChromaDB vector store (deprecated, replaced by FAISS) |

---

### 3.7 📂 `.agents/workflows/` — Developer Runbooks

| File | Purpose |
|------|---------|
| `run-frontend.md` | How to start the React dev server (`npm run dev`) |
| `run-backend-spring.md` | How to start Spring Boot (`mvn spring-boot:run`) |
| `run-backend-fastapi.md` | How to start the Python AI service (`uvicorn main:app`) |
| `run-full-stack.md` | How to run all 3 services together |
| `ingest-documents.md` | How to ingest/re-ingest documents into the FAISS knowledge base |

---

## 🗄️ Database Schema (PostgreSQL RDS)

```
┌──────────────────┐       ┌──────────────────┐
│     users         │       │     folders       │
├──────────────────┤       ├──────────────────┤
│ id (PK)          │◄──┐   │ id (PK)          │
│ email (UNIQUE)   │   │   │ user_id (FK)     │──▶ users
│ password_hash    │   │   │ name             │
│ display_name     │   │   └──────────────────┘
│ username         │   │          ▲
│ avatar_base64    │   │          │
│ is_admin         │   │   ┌──────┴──────────────┐
│ totp_secret      │   │   │      chats          │
└──────────────────┘   │   ├─────────────────────┤
        ▲              │   │ id (PK, UUID)       │
        │              ├──▶│ user_id (FK)        │
        │              │   │ title               │
        │              │   │ preview             │
        │              │   │ pinned              │
        │              │   │ folder_id (FK)      │──▶ folders
        │              │   │ updated_at          │
        │              │   └─────────┬───────────┘
        │              │             │
        │              │   ┌─────────▼───────────┐
        │              │   │     messages         │
        │              │   ├─────────────────────┤
        │              │   │ id (PK)             │
        │              │   │ chat_id (FK)        │──▶ chats
        │              │   │ message (user text) │
        │              │   │ response (AI text)  │
        │              │   │ response_id         │
        │              │   │ response_hash       │
        │              │   │ source_chunk_ids    │
        │              │   │ timestamp           │
        │              │   └───┬─────────┬───────┘
        │              │       │         │
        │              │ ┌─────▼──┐ ┌────▼──────────┐
        │              │ │feedback│ │msg_revisions  │
        │              │ ├────────┤ ├───────────────┤
        │              │ │ id     │ │ id            │
        │              │ │ user   │ │ message_id FK │
        │              │ │ chat_id│ │ response      │
        │              │ │ msg_id │ │ source        │
        │              │ │ type   │ └───────────────┘
        │              │ └────────┘
        │              │
        │    ┌─────────┴──────┐  ┌──────────────┐  ┌───────────────────┐
        └────┤ pending_otps   │  │ prompt_stats  │  │ knowledge_chunks  │
             ├────────────────┤  ├──────────────┤  ├───────────────────┤
             │ id, email      │  │ id           │  │ id                │
             │ otp_code       │  │ prompt_text  │  │ document_id       │
             │ created_at     │  │ count        │  │ file_name         │
             └────────────────┘  │ last_used    │  │ content           │
                                 └──────────────┘  │ token_count       │
                                                   │ embedding         │
             ┌──────────────┐  ┌──────────────────┐└───────────────────┘
             │  documents    │  │ document_chunks  │
             ├──────────────┤  ├──────────────────┤
             │ id           │  │ id               │
             │ file_name    │  │ document_id (FK) │──▶ documents
             │ file_size    │  │ content          │
             │ uploaded_at  │  │ chunk_index      │
             └──────────────┘  └──────────────────┘
```

---

## 🧠 RAG Pipeline — Detailed Technical Breakdown

### Intent Classification

The system classifies every incoming query into one of five intents:

| Intent | Trigger | Behavior |
|--------|---------|----------|
| `GREETING` | "hello", "namaste", "hi" (no procurement keywords) | Returns personalized greeting, **skips RAG entirely** |
| `PROCESS` | Amount-based questions ("₹10 lakh purchase", "Rs. 3 lakh") | Extracts amount → GFR slab → amount-aware search + threshold table injection |
| `ANALYTICAL` | Comparison questions ("LTE vs OTE", "difference between") | Multi-method retrieval, ensures balanced coverage, Markdown table in response |
| `WORKFLOW` | Procedural questions ("how to procure", "what is the SOP") | Full SOP generation, workflow-specific prompt, action-density prioritization |
| `GENERAL` | All other procurement questions | Standard RAG with domain gating |

### Hybrid Retrieval Strategy

```
User Query
    │
    ├─▶ Expand with intent-aware keywords
    │
    ├─▶ Generate query variations (LLM-based OR deterministic)
    │
    ├─▶ For each variation:
    │       ├── Semantic search (FAISS cosine similarity)
    │       └── Keyword search (BM25-style term matching)
    │
    ├─▶ Reciprocal Rank Fusion (RRF, k=60)
    │       • Merges semantic + keyword rankings
    │       • Deduplicates by chunk_id (keeps highest score)
    │
    ├─▶ Domain gating (MIN_PROCUREMENT_SCORE = 2)
    │       • Chunks must contain ≥2 procurement domain terms
    │
    ├─▶ Quality filtering
    │       • Removes TOC pages, scientist name lists, noisy text
    │       • Respects blocked_chunk_ids from feedback loop
    │
    ├─▶ Contextual Compression Rerank
    │       • Flashrank cross-encoder (preferred)
    │       • OR sentence-transformers CrossEncoder
    │       • OR local heuristic scoring fallback
    │
    └─▶ Top-K selection (default K=5)
```

### GFR 2025 Thresholds — Enforced by the System

| Value Band | Procurement Method | GFR Rule | Notes |
|-----------|-------------------|----------|-------|
| Up to ₹50,000 | Direct Purchase | Rule 154 | No quotation needed |
| ₹50,001 – ₹5,00,000 | LPC (Local Purchase Committee) | Rule 155 | Min 3 quotations |
| ₹5,00,001 – ₹25,00,000 | LTE (Limited Tender Enquiry) | Rule 162 | Min 3 firms |
| Above ₹25,00,000 | OTE (Open Tender Enquiry) | Rule 161 | GeM/CPP Portal mandatory |
| Proprietary/Single Source | STE (Single Tender Enquiry) | Rule 166 | PAC certificate mandatory |

> ⚠️ **Anti-Hallucination Guard**: The system forcibly corrects any LLM output that mentions the old "Rs. 2.5 lakh" LTE threshold to the GFR 2025 value of "Rs. 5 lakh".

---

## 🚀 How to Run Locally

```
Terminal 1 — Python AI Service (:8000)
  cd python-ai-service
  ..\venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000

Terminal 2 — Spring Boot (:8080)
  cd backend-spring
  mvn spring-boot:run

Terminal 3 — Frontend (:5173)
  cd frontend
  npm run dev
```

> 💡 **Tip**: The Vite dev server auto-proxies `/api/*` to `:8080` and `/ai/*` to `:8000`, so everything works through `http://localhost:5173`.

---

## 📊 File Count Summary

| Layer | Files | Lines (approx.) |
|-------|-------|-----------------|
| **Frontend** (`frontend/src/`) | 16 source files | ~7,500 |
| **Spring Boot** (`backend-spring/`) | 45 Java files + YAML | ~5,000 |
| **Python AI** (`python-ai-service/app/`) | 10 module files | ~3,400 |
| **Python AI** (legacy monolith) | 1 file (`main_legacy.py`) | ~1,800 (reference only) |
| **Knowledge Base** (`data/`) | 9 PDF documents | ~32 MB corpus |
| **Legacy Backend** (`backend/`) | 8 Python files | ~5,000 (deprecated) |
| **Total active codebase** | **~71 files** | **~15,900 lines** |

---

## 🔑 Key Design Decisions

1. **Split Architecture**: The old monolith was split into Spring Boot (business logic) + Python (AI only) for better scaling and maintainability.

2. **Modularized AI Service**: The 1,800-line monolithic `main.py` was refactored into `app/` package with clean separation: `api/` (routes), `core/` (RAG engine, config, constants, response builder), `services/` (knowledge base, LLM service), `utils/` (processors, text cleaning). The original entry point still works via a thin re-export wrapper.

3. **FAISS over ChromaDB**: Replaced ChromaDB with FAISS for faster, disk-persisted vector search without requiring a separate vector DB server.

4. **Hybrid Retrieval + RRF**: Combines semantic (FAISS cosine) and keyword (BM25-style) search, then fuses rankings with Reciprocal Rank Fusion for robust match quality.

5. **Multi-Stage Reranking**: Three-tier reranking cascade — Flashrank → CrossEncoder → heuristic — ensures the best available reranker is used at runtime.

6. **Agentic RAG Graph**: The pipeline uses a LangGraph-style sequential graph with distinct nodes (question_transformer → multi_query_retrieval → rerank → threshold_logic → generation), making each stage independently testable and extensible.

7. **Sync FastAPI handlers**: The `/chat` endpoint is a sync `def` (not `async def`) because uvicorn/Starlette automatically runs sync handlers in a threadpool (default 40 threads), avoiding event-loop blocking from the CPU-bound FAISS search.

8. **Feedback-aware regeneration**: When a user dislikes a response, the system records the response hash and source chunk IDs. On regeneration, these are passed as `blocked_response_hashes` and `blocked_chunk_ids` to ensure the next answer is genuinely different.

9. **Anti-hallucination**: The system embeds the GFR 2025 threshold table in every prompt AND post-processes the output to correct known hallucination patterns (e.g., old 2.5L LTE threshold → correct 5L).

10. **Seasonal UI**: The frontend detects Indian festivals (Holi, Diwali, Republic Day, etc.) and applies themed particle effects and color schemes for a culturally-aware user experience.
