"""Tests for gin.corpus.db connection helpers."""
from __future__ import annotations

from unittest.mock import patch

import psycopg
import pytest

from gin.corpus.db import DatabaseUnavailableError, connect, ensure_postgres


def test_database_unavailable_error_message():
    exc = DatabaseUnavailableError("postgresql://gin:gin@localhost:5432/gin")
    message = str(exc)
    assert "Could not connect to Postgres" in message
    assert "docker compose up -d" in message
    assert "corpus_ingest.py" in message


def test_ensure_postgres_raises_when_unavailable():
    with patch("gin.corpus.db.postgres_available", return_value=False):
        with pytest.raises(DatabaseUnavailableError) as exc_info:
            ensure_postgres()
    assert "docker compose up -d" in str(exc_info.value)


def test_connect_wraps_operational_error():
    with patch(
        "gin.corpus.db.psycopg.connect",
        side_effect=psycopg.OperationalError("connection refused"),
    ):
        with pytest.raises(DatabaseUnavailableError) as exc_info:
            with connect():
                pass
    assert exc_info.value.cause is not None
    assert "docker compose up -d" in str(exc_info.value)
