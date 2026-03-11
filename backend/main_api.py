"""
main_api.py  —  FastAPI application
====================================
Auth endpoints wired to the merged auth.py:
  POST /api/auth/register/start        → send OTP
  POST /api/auth/register/verify       → verify OTP + create account
  POST /api/auth/login                 → password check, returns must_change + totp_required
  POST /api/auth/change-password       → change password (clears must_change flag)
  POST /api/auth/reset-password        → admin reset → temp password
  POST /api/auth/totp/setup            → generate secret + QR base64
  POST /api/auth/totp/enable           → confirm code, persist secret
  POST /api/auth/totp/verify           → verify live TOTP code during login
  POST /api/auth/totp/disable          → turn off 2FA

Chat endpoints unchanged from v1.
"""

import os
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db, init_db
from backend.models import Message
import backend.auth as auth_service
from backend.core import ask_question

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="CBRI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def home():
    return {"message": "CBRI Chatbot API Running"}


# ── Email helper (configure SMTP env vars to enable real sending) ─────────────

def send_otp_email(email: str, otp: str):
    """
    Send OTP by email.
    Set env vars SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS to enable.
    Falls back to console print (DEBUG) if SMTP is not configured.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not smtp_host or not smtp_user:
        print(f"[DEBUG] OTP for {email}: {otp}")
        return

    msg = MIMEText(
        f"Your CBRI ProcureBuddy registration OTP is:\n\n  {otp}\n\n"
        f"It expires in 10 minutes. Do not share it with anyone."
    )
    msg["Subject"] = "Your Registration OTP — CBRI ProcureBuddy"
    msg["From"]    = smtp_user
    msg["To"]      = email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [email], msg.as_string())
    except Exception as exc:
        # Don't crash the request; OTP still stored in DB
        print(f"[WARN] Failed to send OTP email: {exc}")
        print(f"[DEBUG] OTP for {email}: {otp}")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

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


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/api/auth/register/start")
def register_start(req: RegisterStartRequest, db: Session = Depends(get_db)):
    """Validate domain, create pending OTP record, email OTP to user."""
    ok, result = auth_service.start_registration(db, req.email)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    send_otp_email(req.email, result)   # result is the OTP string on success
    return {"success": True, "message": "OTP sent to your email."}


@app.post("/api/auth/register/verify")
def register_verify(req: RegisterVerifyRequest, db: Session = Depends(get_db)):
    """Verify OTP and create user account with the provided password."""
    ok, msg = auth_service.verify_otp_and_create_user(db, req.email, req.otp, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Password authentication.
    Response includes:
      - must_change  → frontend must show change-password screen
      - totp_required → frontend must ask for TOTP code before granting access
    """
    user = auth_service.authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "success":       True,
        "email":         user.email,
        "must_change":   user.must_change,
        "totp_required": user.totp_enabled,
    }


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, db: Session = Depends(get_db)):
    ok, msg = auth_service.change_password(db, req.email, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Admin endpoint: generate new temp password and return it."""
    ok, result = auth_service.reset_password(db, req.email)
    if not ok:
        raise HTTPException(status_code=404, detail=result)
    return {"success": True, "temp_password": result}


# ── TOTP routes ───────────────────────────────────────────────────────────────

@app.post("/api/auth/totp/setup")
def totp_setup(req: TotpSetupRequest, db: Session = Depends(get_db)):
    """Generate a TOTP secret and QR code (base64 PNG). Secret not saved yet."""
    ok, result = auth_service.generate_totp_setup(db, req.email)
    if not ok:
        raise HTTPException(status_code=404, detail=result)
    return {"success": True, "secret": result["secret"], "qr_base64": result["qr_base64"]}


@app.post("/api/auth/totp/enable")
def totp_enable(req: TotpEnableRequest, db: Session = Depends(get_db)):
    """Confirm TOTP code from authenticator app, then persist secret."""
    ok, msg = auth_service.enable_totp(db, req.email, req.secret, req.code)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/auth/totp/verify")
def totp_verify(req: TotpVerifyRequest, db: Session = Depends(get_db)):
    """Verify a live TOTP code during the login flow."""
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


# ── Chat routes (unchanged) ───────────────────────────────────────────────────

@app.get("/api/chats")
def list_chats(user: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Message.chat_id)
        .filter(Message.user_email == user)
        .distinct()
        .all()
    )
    return {"chat_ids": [r.chat_id for r in rows]}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.user_email == user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    return {
        "chat_id":  chat_id,
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in messages
        ],
    }


@app.post("/api/chats/{chat_id}/message")
def send_message(chat_id: str, req: SendMessageRequest, db: Session = Depends(get_db)):
    # Save user message
    user_msg = Message(
        user_email=req.user,
        chat_id=chat_id,
        role="user",
        content=req.message,
        timestamp=datetime.utcnow(),
    )
    db.add(user_msg)
    db.commit()

    # Build prior history (exclude the message we just saved)
    history_rows = (
        db.query(Message)
        .filter(Message.user_email == req.user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.id != user_msg.id
    ]

    # Call core chatbot
    try:
        reply_text = ask_question(req.message, chat_history)
    except Exception as exc:
        print(f"[ERROR] Chat reply generation failed: {exc}")
        reply_text = (
            "I could not process that request right now because the chatbot backend "
            "is not fully configured. Please check the server logs."
        )

    # Save assistant reply
    bot_msg = Message(
        user_email=req.user,
        chat_id=chat_id,
        role="assistant",
        content=reply_text,
        timestamp=datetime.utcnow(),
    )
    db.add(bot_msg)
    db.commit()

    # Return full updated history
    all_messages = (
        db.query(Message)
        .filter(Message.user_email == req.user, Message.chat_id == chat_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    return {
        "reply":    reply_text,
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in all_messages
        ],
    }
