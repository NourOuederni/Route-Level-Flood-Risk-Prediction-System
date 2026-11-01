"""User accounts: email + password login, sessions, push-token storage, admin role.

Passwords are hashed with PBKDF2-SHA256 (standard library). Sessions are random tokens; only
their SHA-256 hash is stored, so a leaked database does not reveal usable tokens.

First admin (run from backend/):
    python -m app.accounts make-admin someone@example.com
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PBKDF2_ITERATIONS = 200_000
SESSION_DAYS = 30


class AccountBase(DeclarativeBase):
    pass


class User(AccountBase):
    __tablename__ = "app_user"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20), default="user")   # "user" | "admin"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SessionToken(AccountBase):
    __tablename__ = "app_session"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PushToken(AccountBase):
    __tablename__ = "push_token"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("app_user.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AlertEvent(AccountBase):
    __tablename__ = "alert_event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    max_rain_24h_mm: Mapped[float] = mapped_column(Float)
    threshold_mm: Mapped[float] = mapped_column(Float)
    pushes_sent: Mapped[int] = mapped_column(Integer, default=0)


# --- passwords -----------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except ValueError:
        return False


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; PostgreSQL returns aware ones. Treat naive as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --- database holder (set by the app at startup) -------------------------------

class _Store:
    session_factory: sessionmaker | None = None


def init_accounts(db_url: str) -> sessionmaker:
    """Create the account tables and remember the session factory."""
    engine = create_engine(db_url, pool_pre_ping=True)
    AccountBase.metadata.create_all(engine)
    _Store.session_factory = sessionmaker(engine, expire_on_commit=False)
    return _Store.session_factory


def _sf() -> sessionmaker:
    if _Store.session_factory is None:
        raise RuntimeError("accounts not initialised: call init_accounts() at startup")
    return _Store.session_factory


# --- current user ----------------------------------------------------------------

@dataclass(frozen=True)
class CurrentUser:
    id: int
    email: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _issue_session(db, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(SessionToken(token_hash=_token_hash(token), user_id=user.id,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)))
    return token


def current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="login required")
    token = authorization.split(" ", 1)[1].strip()
    with _sf()() as db:
        row = db.execute(select(SessionToken).where(SessionToken.token_hash == _token_hash(token))).scalar_one_or_none()
        if row is None or _aware(row.expires_at) < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="session expired, please log in again")
        user = db.get(User, row.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="unknown user")
        return CurrentUser(id=user.id, email=user.email, role=user.role)


def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin only")
    return user


# --- routes ------------------------------------------------------------------------

router = APIRouter()


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class PushRegistration(BaseModel):
    token: str = Field(min_length=10, max_length=255)


def _public(user: User) -> dict:
    return {"id": user.id, "email": user.email, "role": user.role}


@router.post("/auth/register")
def register(body: Credentials):
    email = body.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=422, detail="enter a valid email address")
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="password must be at least 8 characters")
    with _sf()() as db:
        if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
            raise HTTPException(status_code=409, detail="an account with this email already exists")
        user = User(email=email, password_hash=hash_password(body.password), role="user")
        db.add(user)
        db.flush()
        token = _issue_session(db, user)
        db.commit()
        return {"token": token, "user": _public(user)}


@router.post("/auth/login")
def login(body: Credentials):
    email = body.email.strip().lower()
    with _sf()() as db:
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="wrong email or password")
        token = _issue_session(db, user)
        db.commit()
        return {"token": token, "user": _public(user)}


@router.get("/auth/me")
def me(user: CurrentUser = Depends(current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


@router.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        with _sf()() as db:
            row = db.execute(select(SessionToken).where(SessionToken.token_hash == _token_hash(token))).scalar_one_or_none()
            if row is not None:
                db.delete(row)
                db.commit()
    return {"ok": True}


@router.post("/push/register")
def register_push(body: PushRegistration, user: CurrentUser = Depends(current_user)):
    with _sf()() as db:
        row = db.execute(select(PushToken).where(PushToken.token == body.token)).scalar_one_or_none()
        if row is None:
            db.add(PushToken(token=body.token, user_id=user.id))
        else:
            row.user_id = user.id
        db.commit()
    return {"ok": True}


def push_tokens() -> list[str]:
    with _sf()() as db:
        return [t for (t,) in db.execute(select(PushToken.token)).all()]


def user_count() -> int:
    with _sf()() as db:
        return len(db.execute(select(User.id)).all())


# --- command line: create the first admin -------------------------------------------

def _cli(argv: list[str]) -> int:
    from .db.config import get_db_url

    if len(argv) != 2 or argv[0] != "make-admin":
        print("usage: python -m app.accounts make-admin <email>")
        return 2
    sf = init_accounts(get_db_url())
    email = argv[1].strip().lower()
    with sf() as db:
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            print(f"no account for {email}: register it in the app first, then run this again")
            return 1
        user.role = "admin"
        db.commit()
    print(f"{email} is now an admin")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
