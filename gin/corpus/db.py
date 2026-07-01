"""Postgres connection helpers for the GIN corpus tier."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql://gin:gin@localhost:5432/gin"
_CONNECT_TIMEOUT = 5.0


def database_url() -> str:
    return os.environ.get("GIN_DATABASE_URL", DEFAULT_DATABASE_URL)


def cold_path() -> Path:
    raw = os.environ.get("GIN_COLD_PATH", "data/cold")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


class DatabaseUnavailableError(Exception):
    """Raised when the corpus Postgres instance cannot be reached."""

    def __init__(self, url: str, *, cause: Exception | None = None) -> None:
        self.url = url
        self.cause = cause
        super().__init__(
            f"Could not connect to Postgres at {url}\n"
            "Start the database: cd docker && docker compose up -d\n"
            "Ingest corpus: python scripts/corpus_ingest.py --source data/synthetic"
        )


def postgres_available(connect_timeout: float = 2.0) -> bool:
    try:
        with psycopg.connect(database_url(), connect_timeout=connect_timeout) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def ensure_postgres() -> None:
    if not postgres_available():
        raise DatabaseUnavailableError(database_url())


@contextmanager
def connect():
    url = database_url()
    try:
        with psycopg.connect(url, connect_timeout=_CONNECT_TIMEOUT) as conn:
            yield conn
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailableError(url, cause=exc) from exc


@contextmanager
def transaction():
    with connect() as conn:
        with conn.transaction():
            yield conn
