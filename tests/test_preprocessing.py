import numpy as np
import pandas as pd
import pytest

import preprocessing as pp


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": ["Ann", "Bob", "Bob", "Cy", "Di", "Ed"],
        "city": ["NY", "LA", "LA", None, "NY", None],
        "age": [30.0, 40.0, 40.0, np.nan, 50.0, np.nan],
        "income": [50_000, 60_000, 60_000, 55_000, 1_000_000, np.nan],
    })


def unchanged(original, before):
    pd.testing.assert_frame_equal(original, before)


# ---------- duplicates ----------

def test_remove_duplicates(df, capsys):
    before = df.copy()
    out = pp.remove_duplicates(df)
    assert len(out) == 5
    assert "Removed 1 duplicate" in capsys.readouterr().out
    assert list(out.index) == list(range(5))
    unchanged(df, before)


def test_remove_duplicates_on_subset():
    d = pd.DataFrame({"id": [1, 1, 2], "visit": ["a", "b", "c"]})
    assert len(pp.remove_duplicates(d, subset=["id"], verbose=False)) == 2
    assert pp.remove_duplicates(d, subset=["id"], keep="last", verbose=False)["visit"].tolist() == ["b", "c"]


def test_remove_duplicates_reports_none(capsys):
    pp.remove_duplicates(pd.DataFrame({"a": [1, 2]}))
    assert "No duplicates" in capsys.readouterr().out


# ---------- missing values ----------

def test_fill_missing_defaults(df):
    before = df.copy()
    out = pp.fill_missing(df, verbose=False)
    assert out.isna().sum().sum() == 0
    assert len(out) == 6                       # no row is completely empty
    assert out.loc[3, "age"] == pytest.approx(40.0)   # mean of 30, 40, 40, 50
    assert out.loc[3, "city"] in ("LA", "NY")      # LA and NY tie for most frequent
    unchanged(df, before)


def test_fill_missing_median_and_placeholder(df):
    out = pp.fill_missing(df, numeric="median", categorical="missing", verbose=False)
    assert out.loc[5, "income"] == pytest.approx(60_000)
    assert out.loc[3, "city"] == "Missing"


def test_fill_missing_can_skip_a_type(df):
    out = pp.fill_missing(df, numeric=None, verbose=False)
    assert out["age"].isna().sum() == 2
    assert out["city"].isna().sum() == 0


def test_fill_missing_only_chosen_columns(df):
    out = pp.fill_missing(df, columns=["age"], verbose=False)
    assert out["age"].isna().sum() == 0
    assert out["city"].isna().sum() == 2       # left alone
    assert out["income"].isna().sum() == 1


def test_fill_missing_knn():
    d = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0], "y": [1.0, 2.0, np.nan, 10.0, 11.0]})
    out = pp.fill_missing(d, numeric="knn", n_neighbors=2, verbose=False)
    assert out.loc[2, "y"] == pytest.approx(1.5)   # average of the two nearest rows (x=2, x=1), not 10/11
    assert out["y"].notna().all()


def test_fill_missing_drops_fully_empty_rows():
    d = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
    assert len(pp.fill_missing(d, verbose=False)) == 1


def test_fill_missing_rejects_unknown_method(df):
    with pytest.raises(ValueError):
        pp.fill_missing(df, numeric="average")


# ---------- outliers ----------

def test_find_outliers_iqr(df):
    flags = pp.find_outliers(df, ["income"])
    assert flags["income"].tolist() == [False, False, False, False, True, False]


def test_handle_outliers_remove(df):
    before = df.copy()
    out = pp.handle_outliers(df, ["income"], verbose=False)
    assert len(out) == 5
    assert 1_000_000 not in out["income"].tolist()
    assert out["income"].isna().sum() == 1     # missing values are kept
    unchanged(df, before)


def test_handle_outliers_cap_keeps_rows(df):
    out = pp.handle_outliers(df, ["income"], action="cap", verbose=False)
    assert len(out) == len(df)
    assert out["income"].max() < 1_000_000
    assert out.loc[0, "income"] == 50_000      # normal values untouched


def test_handle_outliers_zscore():
    d = pd.DataFrame({"v": [10.0] * 20 + [100.0]})
    assert len(pp.handle_outliers(d, method="zscore", verbose=False)) == 20


def test_outliers_reject_text_columns(df):
    with pytest.raises(TypeError):
        pp.handle_outliers(df, ["city"])


def test_winsorize():
    d = pd.DataFrame({"v": list(range(1, 101))})
    out = pp.winsorize(d, ["v"], verbose=False)
    assert out["v"].min() == pytest.approx(d["v"].quantile(0.05))
    assert out["v"].max() == pytest.approx(d["v"].quantile(0.95))


def test_log_transform(df):
    out = pp.log_transform(df, ["income"], suffix="_log", verbose=False)
    assert out.loc[0, "income_log"] == pytest.approx(np.log1p(50_000))
    assert "income" in out
    with pytest.raises(ValueError):
        pp.log_transform(pd.DataFrame({"v": [-5.0]}), ["v"])


# ---------- encoding ----------

def test_onehot_encoding(df):
    out = pp.encode_categorical(df, {"city": "onehot"}, verbose=False)
    assert {"city_LA", "city_NY"} <= set(out.columns)
    assert out["city_NY"].tolist() == [1, 0, 0, 0, 1, 0]
    assert list(out.columns).index("city_LA") == list(out.columns).index("city") + 1


def test_label_encoding_keeps_missing(df):
    out = pp.encode_categorical(df, {"city": "label"}, verbose=False)
    assert out["city_label"].tolist()[:3] == [1, 0, 0]       # LA=0, NY=1
    assert out["city_label"].isna().sum() == 2


def test_list_means_onehot_and_drop_original(df):
    out = pp.encode_categorical(df, ["city"], drop_original=True, verbose=False)
    assert "city" not in out and "city_NY" in out


def test_onehot_limit_warns():
    d = pd.DataFrame({"code": [f"c{i}" for i in range(12)]})
    with pytest.warns(UserWarning, match="adds 12 columns"):
        pp.encode_categorical(d, ["code"], verbose=False)


def test_encoding_rejects_bad_input(df):
    with pytest.raises(ValueError):
        pp.encode_categorical(df, {"city": "ordinal"})
    with pytest.raises(KeyError):
        pp.encode_categorical(df, {"country": "label"})


# ---------- normalization ----------

def test_normalize_minmax_and_zscore(df):
    before = df.copy()
    out = pp.normalize(df, {"age": "minmax", "income": "zscore"}, verbose=False)
    assert out["age"].min() == 0 and out["age"].max() == 1
    assert out["age"].isna().sum() == 2
    assert out["income"].mean() == pytest.approx(0, abs=1e-9)
    assert out["income"].std(ddof=0) == pytest.approx(1)
    unchanged(df, before)


def test_normalize_constant_column_is_zero():
    out = pp.normalize(pd.DataFrame({"c": [5, 5, 5]}), ["c"], verbose=False)
    assert out["c"].tolist() == [0, 0, 0]


def test_normalize_rejects_text(df):
    with pytest.raises(TypeError):
        pp.normalize(df, ["city"])


# ---------- age ----------

def test_add_age_detects_dob_and_handles_birthdays():
    d = pd.DataFrame({"patient": [1, 2], "DOB": ["1990-06-15", "1990-06-16"]})
    out = pp.add_age_column(d, as_of="2020-06-15", verbose=False)
    assert out["age"].tolist() == [30, 29]
    assert list(out.columns) == ["patient", "DOB", "age"]


def test_add_age_at_another_date_column():
    d = pd.DataFrame({"birth_date": ["1950-01-01"], "admitted": ["2020-12-31"]})
    out = pp.add_age_column(d, as_of="admitted", name="age_at_admission", verbose=False)
    assert out.loc[0, "age_at_admission"] == 70


def test_add_age_without_dob_column_raises():
    with pytest.raises(KeyError):
        pp.add_age_column(pd.DataFrame({"x": [1]}))


# ---------- pipeline ----------

def test_clean_pipeline(df):
    before = df.copy()
    out = pp.clean(
        df,
        missing={"numeric": "median"},
        outliers={"columns": ["income"], "action": "cap"},
        encode={"columns": {"city": "label"}},
        scale={"columns": {"age": "minmax"}},
        verbose=False,
    )
    assert len(out) == 5                       # one duplicate removed
    assert out.isna().sum().sum() == 0
    assert out["income"].max() < 1_000_000
    assert "city_label" in out
    assert out["age"].between(0, 1).all()
    unchanged(df, before)


def test_clean_requires_columns_for_encode(df):
    with pytest.raises(ValueError, match="encode needs"):
        pp.clean(df, encode=True)
