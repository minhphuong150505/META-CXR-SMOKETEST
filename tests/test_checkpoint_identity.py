import pytest

from smoke.identity import assert_checkpoint_identity, build_checkpoint_identity


VALID = {
    "dataset_manifest_sha256": "b" * 64,
    "config_fingerprint": "c" * 64,
}


def test_identity_requires_all_gating_fingerprints():
    assert build_checkpoint_identity(VALID) == VALID
    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        build_checkpoint_identity({**VALID, "dataset_manifest_sha256": ""})


@pytest.mark.parametrize("field", ("dataset_manifest_sha256", "config_fingerprint"))
def test_resume_rejects_each_identity_mismatch(field):
    actual = {**VALID, field: "different"}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_checkpoint_identity(actual, VALID, "Resume")


def test_resume_ignores_source_commit():
    # A plumbing/code-only commit changes source_commit but leaves the dataset
    # and training config unchanged; the checkpoint must still resume.
    cfg_a = {**VALID, "source_commit": "a" * 40}
    cfg_b = {**VALID, "source_commit": "z" * 40}
    assert build_checkpoint_identity(cfg_a) == build_checkpoint_identity(cfg_b)
    assert_checkpoint_identity(
        build_checkpoint_identity(cfg_a), build_checkpoint_identity(cfg_b), "Resume"
    )
