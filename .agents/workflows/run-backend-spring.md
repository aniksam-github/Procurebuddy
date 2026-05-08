---
description: how to run the Spring Boot (Java) backend
---

## Prerequisites
- Java 17+ installed and on PATH
- Maven (`mvn`) installed and on PATH
- PostgreSQL running locally with a database matching `DB_NAME`, `DB_USER`, `DB_PASSWORD` from `.env`
- `.env` file at `d:\projects\bot\.env` with valid values (especially `DB_*`, `JWT_SECRET`, `SMTP_*`, `GROQ_API_KEY`, `ADMIN_EMAIL`)
- Python 3.11 + venv activated (the Spring Boot backend delegates AI calls to the Python `core.py` via `python_bridge.py`)

## Steps

1. Open a PowerShell terminal and navigate to the Spring Boot backend:
   ```
   cd d:\projects\bot\backend-spring
   ```

2. Run using the provided PowerShell script (reads `.env` automatically and runs `mvn spring-boot:run`):
   ```
   .\run-local.ps1
   ```
   > Alternatively, set env vars manually and run: `mvn spring-boot:run`

3. The server starts on **port 8080** by default. Verify:
   - Open http://localhost:8080/api/health

## Database Setup (First Time)
- Flyway handles migrations automatically on startup.
- Ensure PostgreSQL is running and the DB credentials in `.env` are correct before starting.

## Notes
- Spring Boot depends on the **Python backend** for RAG/AI responses via `python_bridge.py`. Make sure the Python `venv` is activated before running.
- H2 in-memory mode (`procurebuddy-db.mv.db`) is used during local development.
- This backend is a **migration** of the FastAPI backend. The active production backend currently used by the frontend is the FastAPI one (port 8080).
- TOTP (2FA) uses Google Authenticator-compatible QR codes (ZXing + GoogleAuth library).
