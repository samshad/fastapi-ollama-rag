from datetime import datetime
from pathlib import Path

import aiosql
from asyncpg import Record

from fastapi_ollama_rag.core import database

sql_path = Path(__file__).parent.parent / "core" / "sql" / "users.sql"
queries = aiosql.from_path(sql_path, "asyncpg")


async def get_user_by_email(email: str) -> Record | None:
    async with database.pool.acquire() as conn:
        return await queries.get_user_by_email(conn, email=email)


async def create_user(email: str, hashed_password: str) -> Record:
    async with database.pool.acquire() as conn:
        return await queries.create_user(
            conn, email=email, hashed_password=hashed_password
        )


async def save_otp(email: str, code: str, expires_at: datetime) -> None:
    async with database.pool.acquire() as conn:
        await queries.save_otp(conn, email=email, code=code, expires_at=expires_at)


async def get_valid_otp(email: str, code: str) -> Record | None:
    async with database.pool.acquire() as conn:
        return await queries.get_valid_otp(conn, email=email, code=code)


async def delete_otps_for_email(email: str) -> None:
    async with database.pool.acquire() as conn:
        await queries.delete_otps_for_email(conn, email=email)


async def update_user_password(email: str, hashed_password: str) -> None:
    """Overwrites a user's password."""
    async with database.pool.acquire() as conn:
        await queries.update_user_password(
            conn, email=email, hashed_password=hashed_password
        )


async def get_user_by_id(user_id: str) -> Record | None:
    """Fetches a user by their ID."""
    async with database.pool.acquire() as conn:
        return await queries.get_user_by_id(conn, user_id=user_id)
