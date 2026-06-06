"""
Auth Rotaları — /api/v1/auth/*

  POST /auth/register  — Yeni kullanıcı kaydı
  POST /auth/login     — Giriş, access + refresh token döner
  POST /auth/refresh   — Refresh token ile yeni access token al
  GET  /auth/me        — Mevcut kullanıcı bilgileri
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.repositories import UserRepo
from backend.api.auth import (
    UserCreate, UserResponse, Token,
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    decode_token, get_current_user,
)
from jose import JWTError

logger = logging.getLogger("finsight.api.auth")
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """Yeni kullanıcı oluşturur."""
    repo = UserRepo(db)
    existing = await repo.get_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bu e-posta adresi zaten kayıtlı.",
        )
    user = await repo.create(
        email=body.email,
        hashed_pw=hash_password(body.password),
        full_name=body.full_name,
    )
    return user


@router.post("/login", response_model=Token)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """E-posta + şifre ile giriş. Access ve refresh token döner."""
    repo = UserRepo(db)
    user = await repo.get_by_email(form.username)

    if user is None or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-posta veya şifre hatalı.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Hesap devre dışı.")

    await repo.update_last_login(user.id)

    return Token(
        access_token=create_access_token(user.id, user.email),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    """Refresh token ile yeni access token üretir."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz refresh token.",
    )
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise credentials_exc
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise credentials_exc

    repo = UserRepo(db)
    user = await repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exc

    return Token(
        access_token=create_access_token(user.id, user.email),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    """Oturum açmış kullanıcının bilgilerini döner."""
    return current_user
