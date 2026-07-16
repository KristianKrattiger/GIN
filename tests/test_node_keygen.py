"""node_keygen CLI: writes cert+key and prints the fingerprint."""
import subprocess
import sys

from gin.federation.certs import cert_fingerprint


def test_node_keygen_writes_cert_and_prints_fingerprint(tmp_path):
    out_dir = tmp_path / "certs"
    result = subprocess.run(
        [sys.executable, "scripts/node_keygen.py",
         "--node-id", "node_a", "--out-dir", str(out_dir)],
        capture_output=True, text=True, check=True,
    )

    cert_path = out_dir / "node_a" / "cert.pem"
    key_path = out_dir / "node_a" / "key.pem"
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_fingerprint(cert_path) in result.stdout
