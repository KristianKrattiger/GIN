#!/usr/bin/env python3
"""
GIN database table truncate utility.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the local GIN modules can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres, transaction

def wipe_tables() -> None:
    print("[*] Connecting to GIN database...")
    try:
        ensure_postgres()
        with transaction() as conn:
            print("[*] Wiping tables: documents, chunks, edges (CASCADE)...")
            conn.execute("TRUNCATE TABLE documents, chunks, edges CASCADE;")
            print("[+] Tables successfully truncated.")
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[-] Database wipe failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    wipe_tables()