"""Integration tests for optional Matplotlib output."""

from cerberos_peptide_splitter.plots import make_plots
from tests.conftest import requires_mpl


@requires_mpl
def test_plot_bundle_creates_expected_pngs(tmp_path):
    splits = {
        "train": [("a", 1, "ACDEFG"), ("b", 0, "ACDEFA")],
        "val": [("c", 0, "YYYYYY"), ("d", 1, "YYYWYY")],
        "test": [("e", 1, "KLMNQR"), ("f", 0, "KKKKRR")],
    }
    make_plots(splits, str(tmp_path), kmer_k=2)
    expected = {
        "split_sizes.png",
        "label_balance.png",
        "length_distribution.png",
        "biochem_boxplots.png",
        "pca_projection.png",
        "pairwise_similarity.png",
    }
    actual = {path.name for path in (tmp_path / "plots").glob("*.png")}
    assert actual == expected
    assert all((tmp_path / "plots" / name).stat().st_size > 0 for name in expected)


@requires_mpl
def test_plot_bundle_handles_single_record(tmp_path):
    make_plots(
        {"train": [("single", None, "ACDE")], "val": [], "test": []},
        str(tmp_path),
        kmer_k=3,
    )
    assert (tmp_path / "plots" / "split_sizes.png").is_file()
