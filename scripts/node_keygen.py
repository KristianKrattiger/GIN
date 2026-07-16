"""Generate a self-signed identity certificate for one GIN federation node.

Usage:
    python scripts/node_keygen.py --node-id node_a --out-dir certs

Writes certs/<node_id>/{cert.pem,key.pem} and prints the SHA-256 fingerprint
for out-of-band pinning confirmation with peer operators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.federation.certs import cert_fingerprint, generate_self_signed_cert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True, help="this node's node_id")
    parser.add_argument("--out-dir", default="certs", help="output root (default: certs)")
    args = parser.parse_args()

    cert_path, key_path = generate_self_signed_cert(args.node_id, args.out_dir)
    fingerprint = cert_fingerprint(cert_path)

    print(f"[*] wrote {cert_path}")
    print(f"[*] wrote {key_path}")
    print(f"[*] fingerprint: {fingerprint}")
    print("[*] send cert_path to peer operators out-of-band; confirm this "
          "fingerprint matches what they receive before pinning it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
