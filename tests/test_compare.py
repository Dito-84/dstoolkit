"""dst.compare(): has new data drifted from the reference data?"""
import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst


def frame(n: int = 4000, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in rng.integers(0, 10**6, n)],
        "admitted": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "age": rng.normal(60 + shift, 12, n).round(),
        "visits": rng.integers(0, 5, n),
        "ward": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "code": np.where(rng.random(n) < 0.7, rng.choice([f"C{i}" for i in range(10)], n),
                         rng.choice([f"R{i:03d}" for i in range(300)], n)),     # a long tail of rare codes
    })


def row(result: pd.DataFrame, column: str) -> pd.Series:
    return result.set_index("column").loc[column]


def test_same_kind_of_data_is_stable():
    before, after = frame(seed=0), frame(seed=1)
    kept = before.copy()
    result = dst.compare(before, after, verbose=False)
    assert set(result["status"]) == {"stable"}
    assert result["shift_psi"].dropna().max() < 0.05
    pd.testing.assert_frame_equal(before, kept)                           # nothing is changed


def test_long_tail_of_rare_codes_isnt_noise():
    # 300 rare codes: new rare ones appear by chance, and their shares wobble
    r = row(dst.compare(frame(seed=0), frame(seed=1), verbose=False), "code")
    assert r["status"] == "stable" and r["shift_psi"] < 0.05
    assert r["new_categories"]                                            # listed, but not alarming
    assert r["new_category_share"] < 0.02


def test_shifted_numbers():
    big = row(dst.compare(frame(seed=0), frame(seed=1, shift=12), verbose=False), "age")
    assert big["status"] == "major" and big["shift_psi"] >= 0.25 and "distribution shifted" in big["issues"]
    tiny = row(dst.compare(frame(n=20000, seed=0), frame(n=20000, seed=1, shift=0.6), verbose=False), "age")
    assert tiny["p_adjusted"] < 0.05                                      # "significant" with many rows...
    assert tiny["status"] == "stable"                                     # ...but too small to matter


def test_unit_change_and_relabelling_are_major():
    before, after = frame(seed=0), frame(seed=1)
    after["age"] = after["age"] * 12                                      # years -> months
    after["ward"] = after["ward"].replace({"A": "Ward A"})
    result = dst.compare(before, after, verbose=False)
    assert row(result, "age")["status"] == "major"
    ward = row(result, "ward")
    assert ward["status"] == "major" and ward["new_categories"] == ["Ward A"]
    assert set(result["column"][:2]) == {"age", "ward"}                   # the biggest changes come first


@pytest.mark.parametrize(("share", "status"), [(0.10, "major"), (0.03, "moderate"), (0.005, "stable")])
def test_new_categories(share, status):
    before, after = frame(seed=0), frame(seed=1)
    after.loc[after.sample(frac=share, random_state=0).index, "ward"] = "D"
    r = row(dst.compare(before, after, verbose=False), "ward")
    assert r["status"] == status
    assert r["new_categories"] == ["D"] and r["new_category_share"] == pytest.approx(share, abs=0.002)


@pytest.mark.parametrize(("share", "status"), [(0.15, "major"), (0.04, "moderate"), (0.01, "stable")])
def test_more_blanks(share, status):
    before, after = frame(seed=0), frame(seed=1)
    after.loc[after.sample(frac=share, random_state=0).index, "age"] = np.nan
    r = row(dst.compare(before, after, verbose=False), "age")
    assert r["status"] == status
    assert r["blanks_before"] == 0 and r["blanks_after"] == pytest.approx(share, abs=0.002)


def test_type_change_added_and_removed_columns(caplog):
    before, after = frame(seed=0), frame(seed=1)
    after["age"] = after["age"].astype(str)                               # numbers arriving as text
    after = after.drop(columns="visits").assign(bmi=25.0)
    with caplog.at_level("INFO", logger="dstoolkit"):
        result = dst.compare(before, after)
    assert row(result, "age")["status"] == "major" and "type changed: number →" in row(result, "age")["issues"]
    assert row(result, "visits")["status"] == "removed" and row(result, "bmi")["status"] == "added"
    assert "removed: visits" in caplog.text and "added: bmi" in caplog.text


def test_dates_and_identifiers_are_not_compared_but_codes_are():
    before, after = frame(seed=0), frame(seed=1)
    after["admitted"] = after["admitted"] + pd.Timedelta(days=365)        # next year's data: expected
    text_dates = after.assign(admitted=after["admitted"].dt.strftime("%Y-%m-%d"))
    for current in (after, text_dates):
        result = dst.compare(before.assign(admitted=before["admitted"].dt.strftime("%Y-%m-%d"))
                             if current is text_dates else before, current, verbose=False)
        assert row(result, "admitted")["kind"] == "date" and row(result, "admitted")["status"] == "stable"
        assert row(result, "patient_id")["kind"] == "identifier or free text"
        assert row(result, "code")["kind"] == "category"                  # 310 values, but they repeat


def test_whole_numbers_and_decimals_are_the_same_values():
    before = pd.DataFrame({"visits": [0, 1, 2, 3] * 100})
    after = pd.DataFrame({"visits": [0.0, 1.0, 2.0, 3.0] * 99 + [np.nan] * 4})
    r = row(dst.compare(before, after, verbose=False), "visits")
    assert r["status"] == "stable" and r["new_categories"] == []


def test_columns_and_input_errors():
    before, after = frame(seed=0), frame(seed=1)
    assert list(dst.compare(before, after, columns=["ward"], verbose=False)["column"]) == ["ward"]
    with pytest.raises(KeyError, match="not found"):
        dst.compare(before, after, columns=["nope"])
    with pytest.raises(TypeError, match="DataFrame"):
        dst.compare(before, after["age"])
    with pytest.raises(ValueError, match="repeated column"):
        dst.compare(pd.DataFrame([[1, 2]], columns=["a", "a"]), after)
    with pytest.raises(ValueError, match="max_categories"):
        dst.compare(before, after, max_categories=0)
