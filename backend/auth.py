"""
Authentication helpers for ProcureBuddy.
"""

import base64
import random
import secrets
import time
from datetime import datetime
from io import BytesIO

import bcrypt
import pyotp
import qrcode
from sqlalchemy.orm import Session

from models import PendingOTP, User

ALLOWED_DOMAINS = [".cbri@csir.res.in", "@outlook.com", "@gmail.com"]
ADMIN_EMAIL = "aniksam2000@outlook.com"
OTP_EXPIRY_SECS = 600
PASSWORD_MIN_LENGTH = 8


def is_official_email(email: str) -> bool:
    email = email.strip().lower()
    return any(email.endswith(domain) for domain in ALLOWED_DOMAINS)


def is_admin_email(email: str) -> bool:
    return email.strip().lower() == ADMIN_EMAIL


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def generate_temp_password() -> str:
    return secrets.token_urlsafe(8)


def validate_password_strength(password: str) -> tuple[bool, str]:
    password = password or ""

    checks = {
        "length": len(password) >= PASSWORD_MIN_LENGTH,
        "uppercase": any(char.isupper() for char in password),
        "lowercase": any(char.islower() for char in password),
        "digit": any(char.isdigit() for char in password),
        "symbol": any(not char.isalnum() for char in password),
    }

    if all(checks.values()):
        return True, ""

    return (
        False,
        "Password must be at least 8 characters and include uppercase, lowercase, number, and special symbol.",
    )


def generate_otp() -> str:
    return str(random.randint(100000, 999999))


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def start_registration(db: Session, email: str) -> tuple[bool, str]:
    if not is_official_email(email):
        return False, "Please use a CBRI, Outlook, or Gmail email address."

    if get_user_by_email(db, email):
        return False, "An account with this email already exists."

    otp = generate_otp()
    expires_at = time.time() + OTP_EXPIRY_SECS

    pending = db.query(PendingOTP).filter(PendingOTP.email == email).first()
    if pending:
        pending.otp = otp
        pending.expires_at = expires_at
    else:
        pending = PendingOTP(email=email, otp=otp, expires_at=expires_at)
        db.add(pending)

    db.commit()
    return True, otp


def verify_otp_and_create_user(db: Session, email: str, otp: str, password: str) -> tuple[bool, str]:
    pending = db.query(PendingOTP).filter(PendingOTP.email == email).first()

    if not pending:
        return False, "No pending registration for this email."

    if time.time() > pending.expires_at:
        db.delete(pending)
        db.commit()
        return False, "OTP has expired. Please request a new one."

    if otp != pending.otp:
        return False, "Invalid OTP."

    strong_password, password_message = validate_password_strength(password)
    if not strong_password:
        return False, password_message

    user = User(
        email=email,
        password_hash=hash_password(password),
        must_change=False,
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.delete(pending)
    db.commit()
    db.refresh(user)

    return True, "Account created successfully."


def admin_create_user(db: Session, email: str) -> tuple[bool, str]:
    if not is_official_email(email):
        return False, "Please use a CBRI, Outlook, or Gmail email address."

    if get_user_by_email(db, email):
        return False, "User already exists."

    temp_password = generate_temp_password()
    user = User(
        email=email,
        password_hash=hash_password(temp_password),
        must_change=True,
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return True, temp_password


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not check_password(password, user.password_hash):
        return None
    return user


def reset_password(db: Session, email: str) -> tuple[bool, str]:
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    temp_password = generate_temp_password()
    user.password_hash = hash_password(temp_password)
    user.must_change = True
    db.commit()

    return True, temp_password


def change_password(db: Session, email: str, new_password: str) -> tuple[bool, str]:
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    strong_password, password_message = validate_password_strength(new_password)
    if not strong_password:
        return False, password_message

    user.password_hash = hash_password(new_password)
    user.must_change = False
    db.commit()

    return True, "Password changed successfully."


def generate_totp_setup(db: Session, email: str, issuer: str = "CBRI ProcureBuddy"):
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=email, issuer_name=issuer)

    img = qrcode.make(uri)
    buf = BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode()

    return True, {"secret": secret, "qr_base64": qr_b64}


def verify_totp_code(secret: str, code: str) -> bool:
    try:
        return pyotp.TOTP(secret).verify(code)
    except Exception:
        return False


def enable_totp(db: Session, email: str, secret: str, code: str) -> tuple[bool, str]:
    if not verify_totp_code(secret, code):
        return False, "Invalid TOTP code. Please scan again."

    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    user.totp_enabled = True
    user.totp_secret = secret
    db.commit()

    return True, "Two-factor authentication enabled."


def disable_totp(db: Session, email: str) -> tuple[bool, str]:
    user = get_user_by_email(db, email)
    if not user:
        return False, "User not found."

    user.totp_enabled = False
    user.totp_secret = None
    db.commit()

    return True, "Two-factor authentication disabled."


def verify_user_totp(db: Session, email: str, code: str) -> bool:
    user = get_user_by_email(db, email)
    if not user or not user.totp_enabled or not user.totp_secret:
        return False
    return verify_totp_code(user.totp_secret, code)


def is_totp_enabled(db: Session, email: str) -> bool:
    user = get_user_by_email(db, email)
    if not user:
        return False
    return bool(user.totp_enabled)
