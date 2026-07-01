"""Shared pytest fixtures for GIN corpus tests."""
from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest

from gin.corpus.db import database_url, postgres_available


@pytest.fixture(scope="session")
def require_postgres():
    if not postgres_available():
        pytest.skip("Postgres not available — run docker compose in docker/")


@pytest.fixture
def tmp_cold_root(tmp_path, monkeypatch):
    cold = tmp_path / "cold"
    cold.mkdir()
    monkeypatch.setenv("GIN_COLD_PATH", str(cold))
    return cold


@pytest.fixture
def isolated_db(require_postgres, monkeypatch):
    """Use a unique schema per test for isolation."""
    schema = f"test_{uuid.uuid4().hex[:12]}"
    base_url = database_url().split("?")[0]
    monkeypatch.setenv(
        "GIN_DATABASE_URL",
        f"{base_url}?options=-csearch_path%3D{schema}",
    )
    with psycopg.connect(base_url) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        conn.execute(f"SET search_path TO {schema}")
        init_sql = Path(__file__).resolve().parents[1] / "docker" / "init-db.sql"
        conn.execute(init_sql.read_text(encoding="utf-8"))
        conn.commit()
    yield schema
    with psycopg.connect(base_url) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
