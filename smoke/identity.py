from __future__ import annotations


IDENTITY_FIELDS = (
    "source_commit",
    "dataset_manifest_sha256",
    "config_fingerprint",
)


def build_checkpoint_identity(values) -> dict[str, str]:
    identity = {field: values.get(field) for field in IDENTITY_FIELDS}
    missing = [field for field, value in identity.items() if not value]
    if missing:
        raise ValueError("Checkpoint identity is incomplete: " + ", ".join(missing))
    return identity


def assert_checkpoint_identity(actual: dict | None, expected: dict, context: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{context} checkpoint identity mismatch: expected {expected}, found {actual}"
        )
