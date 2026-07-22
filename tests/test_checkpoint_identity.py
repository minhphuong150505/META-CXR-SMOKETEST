import pytest

from smoke.identity import assert_checkpoint_identity, build_checkpoint_identity


VALID = {
    "source_commit": "a" * 40,
    "dataset_manifest_sha256": "b" * 64,
    "config_fingerprint": "c" * 64,
}


def test_identity_requires_all_three_fingerprints():
    assert build_checkpoint_identity(VALID) == VALID
    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        build_checkpoint_identity({**VALID, "dataset_manifest_sha256": ""})


@pytest.mark.parametrize(
    "field", ("source_commit", "dataset_manifest_sha256", "config_fingerprint")
)
def test_resume_rejects_each_identity_mismatch(field):
    actual = {**VALID, field: "different"}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_checkpoint_identity(actual, VALID, "Resume")
