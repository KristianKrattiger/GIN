"""Self-signed cert generation, fingerprinting, and CA-bundle building."""
import os
import ssl

from cryptography import x509

from gin.federation.certs import build_ca_bundle, cert_fingerprint, generate_self_signed_cert


def test_generate_self_signed_cert_writes_expected_paths(tmp_path):
    cert_path, key_path = generate_self_signed_cert("node_a", tmp_path)
    assert cert_path == tmp_path / "node_a" / "cert.pem"
    assert key_path == tmp_path / "node_a" / "key.pem"
    assert cert_path.exists()
    assert key_path.exists()


def test_generate_self_signed_cert_has_expected_common_name(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_b", tmp_path)
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
    assert cn == "node_b"


def test_cert_fingerprint_is_stable_sha256_format(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_a", tmp_path)
    fp = cert_fingerprint(cert_path)
    assert fp.startswith("SHA256:")
    assert len(fp[len("SHA256:"):].split(":")) == 32  # 32-byte digest


def test_cert_fingerprint_differs_for_different_certs(tmp_path):
    cert_a, _ = generate_self_signed_cert("node_a", tmp_path)
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    assert cert_fingerprint(cert_a) != cert_fingerprint(cert_b)


def test_build_ca_bundle_concatenates_all_peer_certs(tmp_path):
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    cert_c, _ = generate_self_signed_cert("node_c", tmp_path)
    bundle = build_ca_bundle([cert_b, cert_c], tmp_path / "bundle.pem")
    assert bundle.read_text() == cert_b.read_text() + cert_c.read_text()


def test_build_ca_bundle_is_usable_as_a_real_ca_store(tmp_path):
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    cert_c, _ = generate_self_signed_cert("node_c", tmp_path)
    bundle = build_ca_bundle([cert_b, cert_c], tmp_path / "bundle.pem")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_verify_locations(cafile=str(bundle))  # must not raise


def test_build_ca_bundle_returns_none_for_empty_peer_list(tmp_path):
    assert build_ca_bundle([], tmp_path / "bundle.pem") is None
    assert not (tmp_path / "bundle.pem").exists()


def test_generate_self_signed_cert_key_file_is_owner_only(tmp_path):
    """The private key is the sole credential once nodes run on shared
    multi-user hosts; it must not be world/group-readable (0644)."""
    _, key_path = generate_self_signed_cert("node_a", tmp_path)
    mode = key_path.stat().st_mode
    if os.name == "posix":
        assert oct(mode)[-3:] == "600"
    else:
        # Windows permission bits don't map to POSIX owner/group/other; the
        # chmod call is a harmless near no-op there. Just confirm the code
        # path that calls chmod(0o600) executed without raising — the file
        # exists and is readable, which it wouldn't be if chmod had errored.
        assert key_path.exists()
