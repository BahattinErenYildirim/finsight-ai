"""
JWT Kimlik Doğrulama — Kullanıcı kayıt, giriş ve token yönetimi.

Kullanılan yöntem: RS256 yerine HS256 (basit kurulum).
Token ömrü: access=30 dk, refresh=7 gün.

.env değişkenleri:
  SECRET_KEY   : Rastgele 32+ karakter (openssl rand -hex 32)
  ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.repositories import UserRepo

logger = logging.getLogger("finsight.auth")

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
_DEFAULT_SECRET = "CHANGE_THIS_IN_PRODUCTION_openssl_rand_hex_32"
SECRET_KEY  = os.getenv("SECRET_KEY", _DEFAULT_SECRET)
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
ENV = os.getenv("ENV", "development").lower()

if SECRET_KEY == _DEFAULT_SECRET:
    msg = "[GÜVENLİK] SECRET_KEY değiştirilmemiş! Üretimde mutlaka .env'e rastgele key ekleyin."
    if ENV in ("production", "prod"):
        raise RuntimeError(msg)
    import warnings
    warnings.warn(msg)

if len(SECRET_KEY) < 32:
    msg = "SECRET_KEY en az 32 karakter olmalıdır."
    if ENV in ("production", "prod"):
        raise RuntimeError(msg)
    logger.warning(msg)

# ── Şifre hasher ─────────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 şeması ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Şema modelleri ────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=120)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isalpha() for c in v) or not any(c.isdigit() for c in v):
            raise ValueError("Şifre en az bir harf ve bir rakam içermelidir.")
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_premium: bool

    model_config = {"from_attributes": True}


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(user_id: int, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "email": email, "exp": expire, "type": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "exp": expire, "type": "refresh"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ── FastAPI dependency'leri ───────────────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Geçerli bir JWT token gerektiren endpoint'ler için dependency."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz veya süresi dolmuş token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise credentials_exc
        user_id: int = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise credentials_exc

    repo = UserRepo(db)
    user = await repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


async def get_current_premium_user(
    current_user=Depends(get_current_user),
):
    """Sadece premium kullanıcılara açık endpoint'ler için."""
    if not current_user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu özellik Premium plana özeldir.",
        )
    return current_user
