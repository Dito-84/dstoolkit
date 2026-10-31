"""associations(): which columns are related, tested once per pair on paired rows."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst
from dstoolkit.exploration import _benjamini_hochberg

DATA = Path(__file__).resolve().parents[1] / "data"


def test_rows_are_paired_before_missing_values_are_dropped():
    # The earlier drafts dropped missing values from each column separately, which
    # pairs values from different rows. Here b is exactly 2 * a on every complete row.
    df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5, 6], "b": [2, np.nan, 6, 8, 10, 12]})
    r = dst.associations(df, method="pearson", significant_only=False, verbose=False)
    assert r.loc[0, "strength"] == pytest.approx(1.0)
    assert r.loc[0, "n"] == 4                       # rows 0, 3, 4, 5


def test_each_pair_is_tested_once():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    r = dst.associations(df, significant_only=False, verbose=False)
    pairs = {frozenset(p) for p in zip(r["column_1"], r["column_2"], strict=True)}
    assert len(r) == len(pairs) == len(list(itertools.combinations("abcde", 2)))


def test_benjamini_hochberg_matches_the_textbook_values():
    adjusted = _benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    assert adjusted.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_unrelated_columns_are_not_reported_as_significant():
    # 190 pairs of pure noise: some raw p-values fall below 0.05 by chance, but none
    # survive the correction for testing many pairs
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(50, 20)), columns=[f"c{i}" for i in range(20)])
    everything = dst.associations(df, significant_only=False, verbose=False)
    assert (everything["p_value"] < 0.05).sum() > 0
    assert dst.associations(df, min_strength=0, verbose=False).empty


def test_method_choice():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    df["y"] = df["x"] + rng.normal(scale=0.5, size=200)
    df["skewed"] = np.exp(df["x"] * 2)
    r = dst.associations(df, significant_only=False, verbose=False).set_index(["column_1", "column_2"])
    assert r.loc[("x", "y"), "test"] == "Pearson"         # both look normal
    assert r.loc[("x", "skewed"), "test"] == "Spearman"   # one is heavily skewed
    assert r.loc[("x", "skewed"), "strength"] == pytest.approx(1.0)   # monotonic, so rho = 1
    forced = dst.associations(df, method="spearman", significant_only=False, verbose=False)
    assert set(forced["test"]) == {"Spearman"}


def test_category_pairs_use_chi_square_and_cramers_v():
    df = pd.DataFrame({"plan": ["basic", "plus"] * 50, "tier": ["low", "high"] * 50,
                       "coin": ["heads", "heads", "tails", "tails"] * 25})
    r = dst.associations(df, significant_only=False, verbose=False).set_index(["column_1", "column_2"])
    assert r.loc[("plan", "tier"), "test"] == "chi-square"
    assert r.loc[("plan", "tier"), "strength"] == pytest.approx(1.0)
    assert r.loc[("plan", "coin"), "strength"] == pytest.approx(0.0)


def test_small_expected_counts_are_flagged():
    df = pd.DataFrame({"a": list("xxxxxxxyyz"), "b": list("ppppppqqqr")})
    r = dst.associations(df, significant_only=False, verbose=False)
    assert "expected counts are below 5" in r.loc[0, "note"]


def test_number_and_category_pairs_use_anova_and_eta():
    df = pd.DataFrame({"group": ["a"] * 30 + ["b"] * 30, "value": [1.0] * 30 + [5.0] * 30})
    df.loc[0, "value"] = 1.5                        # not perfectly constant within a group
    r = dst.associations(df, verbose=False)
    assert r.loc[0, "test"] == "ANOVA" and r.loc[0, "measure"] == "eta"
    assert r.loc[0, "strength"] > 0.99
    tiny = pd.DataFrame({"group": list("aabbccdd") * 2, "value": range(16)})
    note = dst.associations(tiny, significant_only=False, verbose=False).loc[0, "note"]
    assert "fewer than 5 values" in note


def test_ids_free_text_dates_and_constants_are_left_out(capsys):
    df = pd.DataFrame({"id": [f"P{i:03d}" for i in range(60)], "x": range(60),
                       "y": [v * 2 + (v % 3) for v in range(60)],
                       "when": pd.date_range("2020-01-01", periods=60), "same": [1] * 60})
    r = dst.associations(df, max_categories=50)
    assert set(r["column_1"]) | set(r["column_2"]) == {"x", "y"}
    out = capsys.readouterr().out
    assert "Left out: id (60 distinct values), when (dates), same (fewer than 2 distinct values)" in out


def test_titanic_finds_the_known_relationships_and_input_is_unchanged():
    titanic = pd.read_csv(DATA / "titanic3.csv", na_values="?")
    before = titanic.copy()
    r = dst.associations(titanic, verbose=False)
    found = {frozenset(p) for p in zip(r["column_1"], r["column_2"], strict=True)}
    assert {frozenset(("pclass", "fare")), frozenset(("survived", "sex")), frozenset(("sibsp", "parch"))} <= found
    assert (r["p_adjusted"] < 0.05).all() and (r["strength"].abs() >= 0.3).all()
    assert r["strength"].abs().is_monotonic_decreasing
    pd.testing.assert_frame_equal(titanic, before)


def test_bad_arguments():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises(ValueError, match="method"):
        dst.associations(df, method="kendall")
    with pytest.raises(ValueError, match="alpha"):
        dst.associations(df, alpha=5)
    with pytest.raises(KeyError, match="nope"):
        dst.associations(df, columns=["a", "nope"])
