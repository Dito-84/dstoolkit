"""Whole-number identifiers, numeric category order, calendar-day intervals and a data-driven PSI (0.17.0).

Each case is a small made-up frame whose right answer is known.
"""

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst
from dstoolkit.cleaning import _is_numeric_identifier
from dstoolkit.monitoring import _psi

rng = np.random.default_rng(0)


# ---------- recognising whole-number identifiers ----------

@pytest.mark.parametrize("name, values, expected", [
    ("patient_id", [3, 1, 2], True),                               # named like an ID, however few values
    ("PatientID", [10, 20], True),
    ("mrn", [104, 202, 999], True),
    ("encounter", np.arange(1000, 1200), True),                    # 200 distinct, packed: a numbering
    ("cost", rng.integers(1_000, 90_000, 200), False),            # distinct, but spread wide: dollars
    ("visits", rng.integers(0, 6, 200), False),                   # a count
    ("paid", [100, 200, 300], False),                             # "id" inside a word doesn't count
    ("valid", [0, 1, 1], False),
    ("patient_id", [1.5, 2.0, 3.0], False),                       # not whole numbers
    ("age", np.linspace(0, 100, 200), False),
])
def test_numeric_identifier(name, values, expected):
    assert _is_numeric_identifier(pd.Series(values, name=name)) is expected


# ---------- compare(): IDs aren't drift ----------

def two_batches():
    ref = pd.DataFrame({"patient_id": np.arange(1000, 2000), "age": rng.normal(50, 10, 1000)})
    cur = pd.DataFrame({"patient_id": np.arange(2000, 3000), "age": rng.normal(50, 10, 1000)})
    return ref, cur


def test_compare_leaves_a_new_batch_of_ids_alone():
    # before 0.17.0: patient_id was "major", PSI 8.28, on every new batch
    result = dst.compare(*two_batches(), verbose=False).set_index("column")
    assert result.loc["patient_id", "status"] == "stable"
    assert result.loc["patient_id", "kind"] == "identifier or free text"
    assert np.isnan(result.loc["patient_id", "shift_psi"])
    assert "distribution not compared" in result.loc["patient_id", "issues"]


def test_compare_identifiers_can_be_chosen():
    ref, cur = two_batches()
    every = dst.compare(ref, cur, identifiers=[], verbose=False).set_index("column")
    assert every.loc["patient_id", "status"] == "major"          # [] compares every number column
    chosen = dst.compare(ref, cur, identifiers=["age"], verbose=False).set_index("column")
    assert chosen.loc["age", "kind"] == "identifier or free text"
    assert chosen.loc["patient_id", "status"] == "major"         # a list turns the automatic check off
    with pytest.raises(KeyError, match="identifiers"):
        dst.compare(ref, cur, identifiers=["nope"])
    with pytest.raises(TypeError, match="not one string"):
        dst.compare(ref, cur, identifiers="age")


def test_compare_still_flags_blanks_and_type_changes_in_ids():
    ref, cur = two_batches()
    cur = cur.astype({"patient_id": float})
    cur.loc[:199, "patient_id"] = np.nan                            # 20% blank
    row = dst.compare(ref, cur, verbose=False).set_index("column").loc["patient_id"]
    assert row["status"] == "major" and "blanks 0.0% → 20.0%" in row["issues"]


# ---------- profile(): no fill or outlier advice for IDs ----------

def test_profile_gives_numeric_ids_no_fill_or_outlier_advice():
    ids = pd.Series(np.arange(100, 300), dtype=float)
    ids[[3, 7, 11, 19, 25, 40, 41, 42, 60, 70, 80, 90]] = np.nan    # 6% blank
    df = pd.DataFrame({"patient_id": ids, "age": rng.normal(50, 10, 200), "y": rng.integers(0, 2, 200)})
    report = dst.profile(df, charts=False, related=False, target="y")
    advice = [(s.step, s.action) for s in report.suggestions if s.column == "patient_id"]
    assert advice == [("note", None)]                              # was ("missing", "fill_missing"), with the mean
    assert next(c for c in report.columns if c.name == "patient_id").identifier
    assert "patient_id" not in report.pipeline().split("# Left out (identifiers or free text):")[0].split("# 2.")[1]
    assert "identifier, 100 to 299" in report.to_html()


# ---------- category codes in numeric order ----------

def test_label_codes_follow_numeric_order():
    df = pd.DataFrame({"grade": [1, 2, 10, None]})
    out = dst.encode_categorical(df, {"grade": "label"}, verbose=False)
    assert out["grade_label"].tolist()[:3] == [0, 1, 2]           # was [0, 2, 1]: "10" sorted before "2"
    encoder = dst.CategoryEncoder({"grade": "label"}).fit(df)
    assert encoder.categories_["grade"] == [1.0, 2.0, 10.0]
    text = dst.CategoryEncoder({"c": "label"}).fit(pd.DataFrame({"c": ["b", "a", "10", "2"]}))
    assert text.categories_["c"] == ["10", "2", "a", "b"]           # text stays alphabetical


# ---------- DateFeatures: calendar days like add_days_between ----------

def test_interval_units():
    df = pd.DataFrame({"a": pd.to_datetime(["2024-01-01 23:00", "2024-03-09 12:00"]),
                       "b": pd.to_datetime(["2024-01-02 01:00", "2024-03-11 12:00"])})
    elapsed = dst.DateFeatures(columns=[], intervals={"i": ("a", "b")}, drop_original=False).fit_transform(df)
    calendar = dst.DateFeatures(columns=[], intervals={"i": ("a", "b")}, interval_unit="calendar_days",
                                drop_original=False).fit_transform(df)
    assert elapsed["i"].round(3).tolist() == [0.083, 2.0]
    assert calendar["i"].tolist() == [1.0, 2.0]
    assert calendar["i"].tolist() == dst.add_days_between(df, "a", "b", verbose=False)["days_a_to_b"].tolist()
    ny = df.apply(lambda s: s.dt.tz_localize("America/New_York"))
    calendar_ny = dst.DateFeatures(columns=[], intervals={"i": ("a", "b")}, interval_unit="calendar_days",
                                   drop_original=False).fit_transform(ny)
    assert calendar_ny["i"].tolist() == [1.0, 2.0]                 # 47 hours across the change: still 2 days
    with pytest.raises(ValueError, match="interval_unit"):
        dst.DateFeatures(intervals={"i": ("a", "b")}, interval_unit="hours").fit(df)


# ---------- PSI: empty bins set by the data, not a stand-in for zero ----------

def test_psi_by_hand_with_smoothing():
    # 50/50 -> 80/20 with half a row added to each bin: shares 0.5/0.5 and 80.5/101 vs 20.5/101
    a1, a2 = 80.5 / 101, 20.5 / 101
    expected = (a1 - 0.5) * np.log(a1 / 0.5) + (a2 - 0.5) * np.log(a2 / 0.5)
    assert _psi(np.array([50, 50]), np.array([80, 20])) == pytest.approx(expected)
    assert _psi(np.array([30, 70]), np.array([30, 70])) == 0


def test_psi_for_a_brand_new_code_depends_on_the_rows():
    # A new code in half the new rows: the old stand-in (1e-4) gave about 4.6 whatever the size; now the
    # number reflects how surely the reference never had it, and the status is major either way
    small = _psi(np.array([10, 0]), np.array([5, 5]))
    large = _psi(np.array([1000, 0]), np.array([500, 500]))
    assert 0.25 < small < large < 4.6
    ref = pd.DataFrame({"code": ["A"] * 100})
    cur = pd.DataFrame({"code": ["A"] * 50 + ["B"] * 50})
    assert dst.compare(ref, cur, verbose=False)["status"].iloc[0] == "major"
