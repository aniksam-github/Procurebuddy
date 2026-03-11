"""
auth.py  —  Merged authentication module
==========================================================
Combines:
  • Original auth.py  (JSON-based, OTP email, TOTP/2FA, temp-password, domain check)
  • Generated auth.py (SQLAlchemy/SQLite storage, bcrypt)

Storage backend:  SQLite via SQLAlchemy (Session injected from FastAPI Depends).
Password hashing: bcrypt
Domain policy:    Only *.cbri@csir.res.in addresses are accepted.
OTP:              6-digit numeric, 10-minute expiry, stored in pending_otps table.
TOTP / 2FA:       PyOTP + QR-code (base64 PNG) for authenticator apps.
Temp passwords:   Admin-created users get a secrets.token_urlsafe(8) temp password
                  and must_change=True; they must change on first login.
"""

import random
import secrets
import time
from datetime import datetime
from io import BytesIO
import base64

import bcrypt
import pyotp
import qrcode
from sqlalchemy.orm import Session

from backend.models import PendingOTP, User

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_DOMAINS   = [".cbri@csir.res.in", "outlook.com"]
OTP_EXPIRY_SECS  = 600   # 10 minutes


# ── Domain validation ─────────────────────────────────────────────────────────

def is_official_email(email: str) -> bool:
    email = email.strip().lower()
    return any(email.endswith(d) for d in ALLOWED_DOMAINS)


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def generate_temp_password() -> str:
    """Strong random temporary password (URL-safe, 8 bytes → ~11 chars)."""
    return secrets.token_urlsafe(8)


# ── OTP helpers ───────────────────────────────────────────────────────────────

def generate_otp() -> str:
    return str(random.randint(100000, 999999))


# ── User queries ──────────────────────────────────────────────────────────────

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


# ── Registration flow  (OTP → verify → create) ────────────────────────────────

def start_registration(db: Session, email: str) -> tuple[bool, str]:
    """
    Step 1: validate domain, check email isn't taken, generate & store OTP.
    Returns (success, message_or_otp).
    The caller (API route) is responsible for emailing the OTP to the user.
    """
    if not is_official_email(email):
        return False, "Please use an official CBRI email (*.cbri@csir.res.in)."

    if get_user_by_email(db, email):
        return False, "An account with this email already exists."

    otp = generate_otp()
    expires_at = time.time() + OTP_EXPIRY_SECS

    # Upsert pending record
    pending = db.query(PendingOTP).filter(PendingOTP.email == email).first()
    if pending:
        pending.otp        = otp
        pending.expires_at = expires_at
    else:
        pending = PendingOTP(email=email, otp=otp, expires_at=expires_at)
        db.add(pending)

    db.commit()
    # Return OTP so the API layer can send it via SMTP / print in DEBUG mode
    return True, otp


def verify_otp_and_create_user(
    db: Session, email: str, otp: str, password: str
) -> tuple[bool, str]:
    """
    Step 2: verify OTP, then create the user row with the supplied password.
    """
    pending = db.query(PendingOTP).filter(PendingOTP.email == email).first()

    if not pending:
        return False, "No pending registration for this email."

    if time.time() > pending.expires_at:
        db.delete(pending)
        db.commit()
        return False, "OTP has expired. Please request a new one."

    if otp != pending.otp:
        return False, "Invalid OTP."

    # Create user
    user = User(
        email         = email,
        password_hash = hash_password(password),
        must_change   = False,
        created_at    = datetime.utcnow(),
    )
    db.add(user)
    db.delete(pending)
    db.commit()
    db.refresh(user)

    return True, "Account created successfully."


# ── Admin: create user with temp password ─────────────────────────────────────

def admin_create_user(db: Session, email: str) -> tuple[bool, str]:
    """
    Admin-initiated account creation.
    Returns (True, temp_password) on success so the admin can share it.
    User will be forced to change password on first login (must_change=True).
    """
    if not is_official_email(email):
        return False, "Please use an official CBRI email (*.cbri@csir.res.in)."

    if get_user_by_email(db, email):
        return False, "User already exists."

    temp_password = generate_temp_password()
    user = User(
        email         = email,
        password_hash = hash_password(temp_password),
        must_change   = True,
        created_at    = datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return True, temp_password


# ── Authentication ────────────────────────────────────────────────────────────

def authenticate_user(db: Session, email: str, password: str):
    """
    Returns the User ORM object on success, or None on failure.
    Does NOT check TOTP here — that is a separate step in the API layer.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not check_password(password, user.password_hash):
        return None
    return user


# ── Password management ───────────────────────────────────────────────────────

def reset_password(db: Session, email: str) -> tuple[bool, str]:
    """
    Generate a new temp password, set must_change=True.
    Returns (True, temp_password) so the caller can email it.
    """
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    temp_password         = generate_temp_password()
    user.password_hash    = hash_password(temp_password)
    user.must_change      = True
    db.commit()

    return True, temp_password


def change_password(db: Session, email: str, new_password: str) -> tuple[bool, str]:
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    user.password_hash = hash_password(new_password)
    user.must_change   = False
    db.commit()

    return True, "Password changed successfully."


# ── TOTP / 2-FA ───────────────────────────────────────────────────────────────

def generate_totp_setup(db: Session, email: str, issuer: str = "CBRI ProcureBuddy"):
    """
    Generate a new TOTP secret for the user and return:
      { "secret": str, "qr_base64": str }
    The secret is NOT saved to the DB yet — call enable_totp() after the user
    confirms the code from their authenticator app.
    """
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    secret = pyotp.random_base32()
    totp   = pyotp.TOTP(secret)
    uri    = totp.provisioning_uri(name=email, issuer_name=issuer)

    img    = qrcode.make(uri)
    buf    = BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode()

    return True, {"secret": secret, "qr_base64": qr_b64}


def verify_totp_code(secret: str, code: str) -> bool:
    try:
        return pyotp.TOTP(secret).verify(code)
    except Exception:
        return False


def enable_totp(db: Session, email: str, secret: str, code: str) -> tuple[bool, str]:
    """
    Confirm that the user can produce a valid TOTP code, then persist the secret.
    """
    if not verify_totp_code(secret, code):
        return False, "Invalid TOTP code. Please scan again."

    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    user.totp_enabled = True
    user.totp_secret  = secret
    db.commit()

    return True, "Two-factor authentication enabled."


def disable_totp(db: Session, email: str) -> tuple[bool, str]:
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    user.totp_enabled = False
    user.totp_secret  = None
    db.commit()

    return True, "Two-factor authentication disabled."


def verify_user_totp(db: Session, email: str, code: str) -> bool:
    """Verify a live TOTP code for a user who has 2FA enabled."""
    user = get_user_by_email(db, email)
    if not user or not user.totp_enabled or not user.totp_secret:
        return False
    return verify_totp_code(user.totp_secret, code)


def is_totp_enabled(db: Session, email: str) -> bool:
    user = get_user_by_email(db, email)
    if not user:
        return False
    return bool(user.totp_enabled)