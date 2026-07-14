#!/usr/bin/env bash
# Create and populate the two per-node federation databases.
# Prereqs: gin-postgres container running; venv installed.
# Idempotent-ish: CREATE DATABASE fails harmlessly if it already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./venv/Scripts/python.exe

echo "[1/3] creating databases"
docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE gin_node_a OWNER gin;" || true
docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE gin_node_b OWNER gin;" || true

echo "[2/3] applying schema"
docker exec -i gin-postgres psql -U gin -d gin_node_a < docker/init-db.sql
docker exec -i gin-postgres psql -U gin -d gin_node_b < docker/init-db.sql

echo "[3/3] ingesting split corpora (node A <- corpus_node1.json, node B <- corpus_node2.json)"
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_a" \
GIN_COLD_PATH="data/cold_node_a" \
  "$PY" scripts/corpus_ingest.py --source corpus_node1.json --no-edges

GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_b" \
GIN_COLD_PATH="data/cold_node_b" \
  "$PY" scripts/corpus_ingest.py --source corpus_node2.json --no-edges

echo "done. verify:"
docker exec gin-postgres psql -U gin -d gin_node_a -c "SELECT COUNT(*) AS node_a_chunks FROM chunks;"
docker exec gin-postgres psql -U gin -d gin_node_b -c "SELECT COUNT(*) AS node_b_chunks FROM chunks;"
