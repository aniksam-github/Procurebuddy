from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    """
    Registered users.
    Replaces users.json — now persisted in SQLite via SQLAlchemy.
    Carries all fields that the original JSON schema used:
      password_hash, must_change, totp_enabled, totp_secret, created_at.
    """
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String,  unique=True, index=True, nullable=False)
    password_hash = Column(String,  nullable=False)
    must_change   = Column(Boolean, default=False,  nullable=False)
    totp_enabled  = Column(Boolean, default=False,  nullable=False)
    totp_secret   = Column(String,  nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)


class PendingOTP(Base):
    """
    Short-lived OTP records waiting for email verification.
    Replaces pending_users.json.
    Row is deleted once OTP is verified or has expired.
    """
    __tablename__ = "pending_otps"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String,  unique=True, index=True, nullable=False)
    otp        = Column(String,  nullable=False)
    expires_at = Column(Float,   nullable=False)   # Unix timestamp (time.time())


class Message(Base):
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_email = Column(String,  index=True, nullable=False)
    chat_id    = Column(String,  index=True, nullable=False)
    role       = Column(String,  nullable=False)   # "user" | "assistant"
    content    = Column(Text,    nullable=False)
    timestamp  = Column(DateTime, default=datetime.utcnow)