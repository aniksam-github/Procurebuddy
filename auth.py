import hashlib
import json
import os
import random
import time

import bcrypt
import secrets
from datetime import datetime

USERS_FILE = "users.json"
PENDING_FILE = "pending_users.json"
OTP_EXPIRY_SECONDS = 600 # 10 mins
ALLOWED_DOMAINS = [".cbri@csir.res.in"]

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def is_official_email(email):
    return email.lower().endswith(".cbri@csir.res.in")

def generate_otp():
    return str(random.randint(100000,999999))

def send_otp_email():
    # TODO: configure SMTP (we'll wire this next)
    print("DEBUG OTP for", email, ":", otp)


def start_create_account(email):
    if not is_official_email(email):
        return False, "please use official CBRI email."

    users = load_json(USERS_FILE)
    if email in users:
        return False, "Account already exists."

    pending = load_json(PENDING_FILE)

    otp = generate_otp()
    pending[email] = {
        "otp" : otp,
        "expires_at": time.time() + OTP_EXPIRY_SECONDS
    }

    save_json(PENDING_FILE, pending)
    send_otp_email(email, otp)

    return True, "OTP sent to your email. "

def verify_otp_and_create_user(email, otp, password):
    pending = load_json(PENDING_FILE)

    if email not in pending:
        return False, "No pending request for this email."

    record = pending[email]

    if time.time() > record["expires_at"]:
        del pending[email]
        save_json(PENDING_FILE, pending)
        return False, "OTP expired. please try again. "

    if otp!= record["otp"]:
        return False, "Invalid OTP."

    users = load_json(USERS_FILE)

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    users[email] = {
        "password_hash":password_hash,
        "must_change": False
    }

    save_json(USERS_FILE, users)

    del pending[email]
    save_json(PENDING_FILE, pending)

    return True, "Account created successfully."


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}   # ✅ ALWAYS return dict



def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users,f, indent=2)

def is_official_email(email: str):
    email = email.lower()
    return any(email.endswith(d) for d in ALLOWED_DOMAINS)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

def generate_temp_password():
    # strong random password
    return secrets.token_urlsafe(8)

def create_user(email):
    if not is_official_email(email):
        return False, "Please use official CBRI email."

    users = load_users()
    if users is None:
        users = {}

    if email in users:
        return False, "User already exists."

    temp_password = generate_temp_password()
    users[email] = {
        "password_hash" : hash_password(temp_password),
        "must_change" : True,
        "created_at" : datetime.now().isoformat()
    }

    save_users(users)
    return True, temp_password

def authenticate_user(email:str, password: str):
    users = load_users()

    if email not in users:
        return False, "User not found."

    user = users[email]
    if not check_password(password, user["password_hash"]):
        return False, "Invalid password."

    return True, user

def reset_password(email: str):
    users = load_users()

    if email not in users:
        return False, "User not found."

    temp_password = generate_temp_password()
    users[email]["password_hash"] = hash_password(temp_password)
    users[email]["must_change"] = True

    save_users(users)
    return True, temp_password

def change_password(email: str, new_password: str):
    users = load_users()

    if email not in users:
        return False, "User not found."

    users[email]["password_hash"] = hash_password(new_password)
    users[email]["must_change"] = False

    save_users(users)
    return True, "Password changed successfully."