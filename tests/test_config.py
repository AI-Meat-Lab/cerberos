"""Validation tests for the public run configuration."""

import pytest

from cerberos_peptide_splitter.config import RunConfig


def test_default_config_is_normalized_and_json_friendly():
    config = RunConfig()
    assert config.train_pct + config.val_pct + config.test_pct == pytest.approx(1.0)
    assert config.as_dict()["kmer_sizes"] == [3]
    assert config.as_dict()["max_candidate_pairs"] == 250_000


def test_percentages_accept_common_scale_and_normalize():
    config = RunConfig(train_pct=80, val_pct=10, test_pct=10)
    assert (config.train_pct, config.val_pct, config.test_pct) == pytest.approx(
        (0.8, 0.1, 0.1)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_pct": 0, "val_pct": 0, "test_pct": 0},
        {"train_pct": -0.1, "val_pct": 0.5, "test_pct": 0.6},
        {"kmer_sizes": []},
        {"kmer_sizes": [0]},
        {"num_hashes": 3},
        {"similarity_threshold": 1.1},
        {"prefilter_threshold": -0.1},
        {"seed": -1},
        {"seed": True},
        {"train_pct": True},
        {"no_clustering": 1},
        {"max_candidate_pairs": 0},
        {"max_candidate_pairs": True},
        {"seed": 2**32},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


def test_prefilter_cannot_be_tighter_than_final_threshold():
    with pytest.raises(ValueError, match="prefilter_threshold"):
        RunConfig(
            verify_with="levenshtein",
            similarity_threshold=0.7,
            prefilter_threshold=0.8,
        )
