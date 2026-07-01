"""Hot tier — dense embeddings stored in pgvector."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector

from . import warm

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def register(conn: psycopg.Connection) -> None:
    register_vector(conn)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    vectors = _model().encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def embed_and_store(conn: psycopg.Connection, chunk_id: str, text: str) -> None:
    register(conn)
    vector = embed_query(text)
    warm.set_chunk_embedding(conn, chunk_id, vector)
