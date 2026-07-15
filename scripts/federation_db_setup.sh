#!/usr/bin/env bash
# Create and populate the three per-node federation databases.
# Prereqs: gin-postgres container running; venv installed.
# Idempotent-ish: CREATE DATABASE fails harmlessly if it already exists;
# init-db.sql is all IF NOT EXISTS so re-applying only adds new tables.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./venv/Scripts/python.exe

echo "[1/3] creating databases"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE $db OWNER gin;" || true
done

echo "[2/3] applying schema (adds peer_anchors + peer_summaries if missing)"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec -i gin-postgres psql -U gin -d "$db" < docker/init-db.sql
done

echo "[3/3] ingesting split corpora"
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_a" \
GIN_COLD_PATH="data/cold_node_a" "$PY" scripts/corpus_ingest.py --source corpus_node1.json --no-edges
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_b" \
GIN_COLD_PATH="data/cold_node_b" "$PY" scripts/corpus_ingest.py --source corpus_node2.json --no-edges
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_c" \
GIN_COLD_PATH="data/cold_node_c" "$PY" scripts/corpus_ingest.py --source corpus_node3.json --no-edges

echo "done. verify:"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec gin-postgres psql -U gin -d "$db" -c "SELECT '$db' AS db, COUNT(*) AS chunks FROM chunks;"
done
