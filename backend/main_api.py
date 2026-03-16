"""
FastAPI application for the ProcureBuddy frontend and backend.
"""

import os
import smtplib
import shutil
import threading
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth as auth_service
from core import ask_question
from database import get_db, init_db
from ingest import SUPPORTED_DOC_EXTENSIONS, create_vector_db
from models import Message

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=PROJECT_ROOT / "backend" / ".env")

app = FastAPI(title="CBRI ProcureBuddy API")


def _split_origins(value: str) -> list[str]:
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


def _cors_options() -> dict:
    configured = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    origin_regex = os.getenv("CORS_ALLOWED_ORIGIN_REGEX", "").strip() or None

    if configured:
        origins = _split_origins(configured)
        if "*" in origins:
            # This API does not use cookie-based auth, so wildcard origins are safe
            # as long as credentials are not enabled.
            return {
                "allow_origins": ["*"],
                "allow_credentials": False,
                "allow_origin_regex": None,
            }
        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_origin_regex": origin_regex,
        }

    # Safe defaults for split frontend/backend deployments.
    return {
        "allow_origins": ["*"],
        "allow_credentials": False,
        "allow_origin_regex": origin_regex,
    }


_cors = _cors_options()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors["allow_origins"],
    allow_origin_regex=_cors["allow_origin_regex"],
    allow_credentials=_cors["allow_credentials"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESS_LOCK = threading.Lock()
PROCESS_STATE = {
    "busy": False,
    "stage": "idle",
    "started_at": None,
    "finished_at": None,
    "last_result": None,
    "last_error": None,
}


class RegisterStartRequest(BaseModel):
    email: str


class RegisterVerifyRequest(BaseModel):
    email: str
    otp: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    email: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    email: str


class TotpSetupRequest(BaseModel):
    email: str


class TotpEnableRequest(BaseModel):
    email: str
    secret: str
    code: str


class TotpVerifyRequest(BaseModel):
    email: str
    code: str


class TotpDisableRequest(BaseModel):
    email: str


class SendMessageRequest(BaseModel):
    user: str
    message: str


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def home():
    return {"message": "CBRI ProcureBuddy API is running."}


@app.get("/api/health")
def health():
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}


def send_otp_email(email: str, otp: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not smtp_host or not smtp_user or not smtp_pass:
        raise RuntimeError(
            "OTP email is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASS before registering."
        )

    msg = MIMEText(
        f"Your ProcureBuddy registration OTP is:\n\n{otp}\n\n"
        "It expires in 10 minutes. Do not share it with anyone."
    )
    msg["Subject"] = "ProcureBuddy registration OTP"
    msg["From"] = smtp_user
    msg["To"] = email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [email], msg.as_string())
    except Exception as exc:
        raise RuntimeError(f"Failed to send OTP email to {email}.") from exc


def _chat_title_from_messages(messages: list[Message]) -> str:
    for message in messages:
        if message.role == "user" and message.content.strip():
            return message.content.strip().replace("\n", " ")[:60]
    for message in messages:
        if message.content.strip():
            return message.content.strip().replace("\n", " ")[:60]
    return "New Chat"


def _chat_preview_from_messages(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.content.strip():
            return message.content.strip().replace("\n", " ")[:120]
    return ""


def _build_chat_summaries(messages: list[Message]) -> list[dict]:
    grouped: dict[str, list[Message]] = {}
    for message in messages:
        grouped.setdefault(message.chat_id, []).append(message)

    summaries = []
    for chat_id, chat_messages in grouped.items():
        latest = chat_messages[-1]
        summaries.append(
            {
                "chat_id": chat_id,
                "title": _chat_title_from_messages(chat_messages),
                "preview": _chat_preview_from_messages(chat_messages),
                "message_count": len(chat_messages),
                "updated_at": latest.timestamp.isoformat(),
            }
        )

    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def _format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _set_process_state(**changes):
    PROCESS_STATE.update(changes)


def _require_admin(email: str):
    if not email or not auth_service.is_admin_email(email):
        raise HTTPException(status_code=403, detail="Admin access is restricted to the configured admin account.")


def _run_processing_cycle(trigger: str):
    if not PROCESS_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Document processing is already running.")

    _set_process_state(
        busy=True,
        stage=f"{trigger}: preparing",
        started_at=datetime.utcnow().isoformat(),
        finished_at=None,
        last_error=None,
    )

    try:
        _set_process_state(stage=f"{trigger}: OCR, chunking, and vector refresh")
        result = create_vector_db()
        finished_at = datetime.utcnow().isoformat()
        _set_process_state(
            busy=False,
            stage="idle",
            finished_at=finished_at,
            last_result={**result, "trigger": trigger, "finished_at": finished_at},
        )
        return result
    except Exception as exc:
        finished_at = datetime.utcnow().isoformat()
        _set_process_state(
            busy=False,
            stage="idle",
            finished_at=finished_at,
            last_error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PROCESS_LOCK.release()


@app.post("/api/auth/register/start")
def register_start(req: RegisterStartRequest, db: Session = Depends(get_db)):
    ok, result = auth_service.start_registration(db, req.email)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    try:
        send_otp_email(req.email, result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": True, "message": "OTP sent to your email."}


@app.post("/api/auth/register/verify")
def register_verify(req: RegisterVerifyRequest, db: Session = Depends(get_db)):
    ok, msg = auth_service.verify_otp_and_create_user(db, req.email, req.otp, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "success": True,
        "email": user.email,
        "must_change": user.must_change,
        "totp_required": user.totp_enabled,
        "totp_enabled": user.totp_enabled,
        "is_admin": auth_service.is_admin_email(user.email),
    }


@app.get("/api/auth/status")
def auth_status(email: str, db: Session = Depends(get_db)):
    user = auth_service.get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "success": True,
        "email": user.email,
        "must_change": user.must_change,
        "totp_enabled": user.totp_enabled,
        "is_admin": auth_service.is_admin_email(user.email),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, db: Session = Depends(get_db)):
    ok, msg = auth_service.change_password(db, req.email, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    ok, result = auth_service.reset_password(db, req.email)
    if not ok:
        raise HTTPException(status_code=404, detail=result)
    return {"success": True, "temp_password": result}


@app.post("/api/auth/totp/setup")
def totp_setup(req: TotpSetupRequest, db: Session = Depends(get_db)):
    ok, result = auth_service.generate_totp_setup(db, req.email)
    if not ok:
        raise HTTPException(status_code=404, detail=result)
    return {"success": True, "secret": result["secret"], "qr_base64": result["qr_base64"]}


@app.post("/api/auth/totp/enable")
def totp_enable(req: TotpEnableRequest, db: Session = Depends(get_db)):
    ok, msg = auth_service.enable_totp(db, req.email, req.secret, req.code)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/totp/verify")
def totp_verify(req: TotpVerifyRequest, db: Session = Depends(get_db)):
    valid = auth_service.verify_user_totp(db, req.email, req.code)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or expired TOTP code.")
    return {"success": True, "message": "TOTP verified."}


@app.post("/api/auth/totp/disable")
def totp_disable(req: TotpDisableRequest, db: Session = Depends(get_db)):
    ok, msg = auth_service.disable_totp(db, req.email)
    if not ok:
        raise HTTPException(status_code=404, detail=msg)
    return {"success": True, "message": msg}


@app.get("/api/chats")
def list_chats(user: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.user_email == user)
        .order_by(Message.timestamp.asc())
        .all()
    )
    chats = _build_chat_summaries(messages)
    return {"chat_ids": [chat["chat_id"] for chat in chats], "chats": chats}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.user_email == user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    return {
        "chat_id": chat_id,
        "title": _chat_title_from_messages(messages) if messages else "New Chat",
        "messages": [
            {"role": message.role, "content": message.content, "timestamp": message.timestamp.isoformat()}
            for message in messages
        ],
    }


@app.post("/api/chats/{chat_id}/message")
def send_message(chat_id: str, req: SendMessageRequest, db: Session = Depends(get_db)):
    if PROCESS_STATE["busy"]:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base update in progress. Chat is temporarily paused until processing completes.",
        )
    if not req.user.strip():
        raise HTTPException(status_code=400, detail="User email is required.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    user_message = Message(
        user_email=req.user,
        chat_id=chat_id,
        role="user",
        content=req.message.strip(),
        timestamp=datetime.utcnow(),
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    history_rows = (
        db.query(Message)
        .filter(Message.user_email == req.user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    chat_history = [
        {"role": row.role, "content": row.content}
        for row in history_rows
        if row.id != user_message.id
    ]

    try:
        reply_text = ask_question(req.message.strip(), chat_history)
    except Exception as exc:
        print(f"[ERROR] Chat reply generation failed: {exc}")
        reply_text = (
            "I could not process that request right now because the chatbot backend "
            "is not fully configured. Please check the server logs."
        )

    bot_message = Message(
        user_email=req.user,
        chat_id=chat_id,
        role="assistant",
        content=reply_text,
        timestamp=datetime.utcnow(),
    )
    db.add(bot_message)
    db.commit()

    all_messages = (
        db.query(Message)
        .filter(Message.user_email == req.user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    return {
        "reply": reply_text,
        "chat": {
            "chat_id": chat_id,
            "title": _chat_title_from_messages(all_messages),
            "preview": _chat_preview_from_messages(all_messages),
            "message_count": len(all_messages),
            "updated_at": all_messages[-1].timestamp.isoformat() if all_messages else None,
        },
        "messages": [
            {"role": message.role, "content": message.content, "timestamp": message.timestamp.isoformat()}
            for message in all_messages
        ],
    }


@app.get("/api/admin/documents")
def list_documents(email: str):
    _require_admin(email)
    documents = []
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
                continue
            stat = path.stat()
            documents.append(
                {
                    "name": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size_bytes": stat.st_size,
                    "size_label": _format_size(stat.st_size),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    return {"success": True, "documents": documents, "count": len(documents)}


@app.get("/api/admin/status")
def admin_status(email: str):
    _require_admin(email)
    return {"success": True, **PROCESS_STATE}


@app.post("/api/admin/upload")
async def upload_documents(email: str, files: list[UploadFile] = File(...)):
    _require_admin(email)
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one document to upload.")

    uploaded = []
    for file in files:
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()
        if not filename or extension not in SUPPORTED_DOC_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file '{file.filename}'. Allowed: {', '.join(sorted(SUPPORTED_DOC_EXTENSIONS))}",
            )

        destination = DATA_DIR / filename
        with destination.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        uploaded.append(filename)

    result = _run_processing_cycle("upload")
    return {
        "success": True,
        "message": "Documents uploaded and knowledge base refreshed successfully.",
        "uploaded": uploaded,
        **result,
    }


@app.post("/api/admin/reindex")
def reindex_documents(email: str):
    _require_admin(email)
    result = _run_processing_cycle("reindex")
    return {
        "success": True,
        "message": "Knowledge base reindexed successfully.",
        **result,
    }
