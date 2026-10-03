"""Tests for cerberos.minhash."""
from __future__ import annotations

import numpy as np

from cerberos.kmers import jaccard, kmers
from cerberos.minhash import (
    estimate_jaccard,
    lsh_candidates,
    minhash_sketch,
)


# ─────────────────────── sketch ───────────────────────

def test_sketch_shape_dtype():
    sk = minhash_sketch({"AA", "AC"}, num_hashes=16, seed=1)
    assert sk.shape == (16,)
    assert sk.dtype == np.int64


def test_sketch_deterministic():
    s = {"AA", "AC", "AG"}
    a = minhash_sketch(s, 32, 7)
    b = minhash_sketch(s, 32, 7)
    assert np.array_equal(a, b)


def test_sketch_seed_changes_values():
    s = {"AA", "AC", "AG"}
    a = minhash_sketch(s, 32, 1)
    b = minhash_sketch(s, 32, 2)
    assert not np.array_equal(a, b)


def test_sketch_empty_is_prime():
    sk = minhash_sketch(set(), 8, 0)
    assert np.all(sk == 2147483647)


def test_sketch_estimates_jaccard_within_tolerance():
    rng = np.random.RandomState(0)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    a = {"".join(rng.choice(alphabet, 3)) for _ in range(400)}
    b = set(list(a)[:200]) | {
        "".join(rng.choice(alphabet, 3)) for _ in range(200)
    }
    true_j = jaccard(a, b)
    sk_a = minhash_sketch(a, 512, 3)
    sk_b = minhash_sketch(b, 512, 3)
    est = estimate_jaccard(sk_a, sk_b)
    assert abs(est - true_j) < 0.10


# ─────────────────────── LSH ───────────────────────

def test_lsh_finds_near_duplicates():
    sets = [
        {"AA", "AC", "AG"},
        {"AA", "AC", "AG"},
        {"ZZ", "YY", "XX"},
    ]
    sketches = np.array([minhash_sketch(s, 64, 0) for s in sets])
    pairs = lsh_candidates(sketches, num_hashes=64, rows=4)
    assert (0, 1) in pairs


def test_lsh_no_far_pair():
    sets = [
        {"AA", "AC", "AG"},
        {"ZZ", "YY", "XX"},
    ]
    sketches = np.array([minhash_sketch(s, 64, 0) for s in sets])
    pairs = lsh_candidates(sketches, num_hashes=64, rows=4)
    assert (0, 1) not in pairs


def test_lsh_singletons_no_pairs():
    sketches = np.array([minhash_sketch({"AA"}, 16, 0),
                         minhash_sketch({"ZZ"}, 16, 0)])
    pairs = lsh_candidates(sketches, 16, rows=4)
    assert pairs == set()


# ─────────────────────── estimate_jaccard ───────────────────────

def test_estimate_identical_sketch_is_one():
    sk = minhash_sketch({"AA", "AC"}, 32, 0)
    assert estimate_jaccard(sk, sk) == 1.0


def test_estimate_symmetric():
    a = minhash_sketch({"AA", "AC"}, 32, 0)
    b = minhash_sketch({"AC", "AG"}, 32, 0)
    assert estimate_jaccard(a, b) == estimate_jaccard(b, a)
