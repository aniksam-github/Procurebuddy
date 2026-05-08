---
description: how to run the React + Vite frontend
---

## Prerequisites
- Node.js 18+ and npm installed
- The backend (FastAPI or Spring Boot) must be running on **port 8080**
- `VITE_API_BASE_URL` is set to `http://localhost:8080` in `.env` at the project root (already set by default)

## Steps

1. Open a terminal and navigate to the frontend directory:
   ```
   cd d:\projects\bot\frontend
   ```

2. (First time only) Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open the app in your browser:
   - http://localhost:5173

## Notes
- Vite's dev server proxies all `/api` requests to `http://127.0.0.1:8080`, so CORS is handled automatically in development.
- Tech stack: React 18, Vite 5, Tailwind CSS 3, Framer Motion, React Router v6, Axios, React Markdown.
- To build for production:
  ```
  npm run build
  ```
  The output will be in `frontend/dist/`.

## Key Pages / Routes
| Route | Component | Description |
|---|---|---|
| `/` | `Home.jsx` | Landing page with animations |
| `/login` | `LoginPage.jsx` | Login / Register / OTP / TOTP flow |
| `/chat` | `Views.jsx` | Main chatbot interface |
| `/sidebar` | `Sidebar.jsx` | Chat history sidebar |
