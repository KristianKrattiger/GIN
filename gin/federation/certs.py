"""Self-signed peer certificates for mutual TLS.

Each node is its own certificate authority: it presents a self-signed cert
as its identity and trusts only the specific certificates operators have
pinned for each peer (docs/superpowers/specs/2026-07-16-federation-mtls-design.md).
No CA, no shared secret — the pinned certificate file IS the trust anchor.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_VALIDITY_DAYS = 3650


def generate_self_signed_cert(node_id: str, certs_root: str | Path) -> tuple[Path, Path]:
    """Write a self-signed ECDSA P-256 cert+key for node_id under
    certs_root/node_id/{cert.pem,key.pem}. Returns (cert_path, key_path).

    BasicConstraints(ca=True) and KeyUsage(key_cert_sign=True) are required
    (and marked critical) for this self-signed cert to work as its own trust
    anchor when pinned directly into a peer's CA bundle — without them,
    OpenSSL's chain validation rejects it even when the exact cert is
    present in the trust store.
    """
    out_dir = Path(certs_root) / node_id
    out_dir.mkdir(parents=True, exist_ok=True)

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                key_encipherment=False, content_commitment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def cert_fingerprint(cert_path: str | Path) -> str:
    """SHA-256 fingerprint of the cert at cert_path, for out-of-band pinning
    confirmation (e.g. "SHA256:ab:cd:...")."""
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    digest = cert.fingerprint(hashes.SHA256())
    return "SHA256:" + ":".join(f"{b:02x}" for b in digest)


def build_ca_bundle(
    peer_cert_paths: Sequence[str | Path], bundle_path: str | Path
) -> Path | None:
    """Concatenate pinned peer certs into one CA bundle file for
    ssl_ca_certs. Returns None (writes nothing) if peer_cert_paths is empty
    — an empty bundle file is invalid and crashes
    SSLContext.load_verify_locations with CERTIFICATE_VERIFY_FAILED /
    NO_CERTIFICATE_OR_CRL_FOUND; callers must skip ssl_ca_certs/CERT_REQUIRED
    entirely in that case rather than pass an empty file.
    """
    if not peer_cert_paths:
        return None
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(Path(p).read_text(encoding="utf-8") for p in peer_cert_paths)
    bundle_path.write_text(contents, encoding="utf-8")
    return bundle_path
