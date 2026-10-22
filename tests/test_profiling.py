import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst
from dstoolkit.profiling import Profile


@pytest.fixture
def messy():
    rng = np.random.default_rng(0)
    n = 200
    income = rng.lognormal(10, 1, n)                           # long right tail
    df = pd.DataFrame({
        "id": [f"P{i:04d}" for i in range(n)],                 # identifier
        "age": rng.normal(45, 12, n).round(),
        "income": income,
        "visits": rng.integers(0, 4, n),                       # small counts
        "sex": rng.choice(["F", "M"], n),
        "plan": rng.choice(["Basic", "Plus", "Premium"], n),
        "zip_text": [str(z) for z in rng.integers(10000, 99999, n)],  # numbers stored as text
        "notes": [None] * 150 + ["follow up"] * 50,            # mostly empty
        "country": ["US"] * n,                                 # constant
        "date_of_birth": pd.date_range("1950-01-01", periods=n, freq="70D").strftime("%Y-%m-%d"),
        "blank": [np.nan] * n,                                 # empty
    })
    df.loc[:19, "age"] = np.nan                                # 10% missing
    df.loc[:4, "plan"] = None                                  # 2.5% missing
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)   # 2 duplicate rows
    return df


def by_column(report, column):
    return [s for s in report.suggestions if s.column == column]


def test_profile_does_not_modify_input(messy):
    before = messy.copy()
    dst.profile(messy, charts=False)
    pd.testing.assert_frame_equal(messy, before)


def test_overview_numbers(messy):
    r = dst.profile(messy, charts=False)
    assert (r.n_rows, r.n_columns, r.duplicates) == (202, 11, 2)
    assert r.missing_cells == int(messy.isna().sum().sum())


def test_column_kinds(messy):
    kinds = dst.profile(messy, charts=False).summary()["type"].to_dict()
    assert kinds["id"] == "text"
    assert kinds["age"] == "numeric"
    assert kinds["sex"] == "categorical"
    assert kinds["date_of_birth"] == "datetime"
    assert kinds["blank"] == "empty"


def test_suggestions_cover_each_issue(messy):
    r = dst.profile(messy, charts=False)
    actions = {(s.column, s.action) for s in r.suggestions}
    assert (None, "remove_duplicates") in actions
    assert ("blank", "drop") in actions
    assert ("country", "drop") in actions                  # constant
    assert ("notes", "drop") in actions                    # 75% missing
    assert ("zip_text", "to_numeric") in actions
    assert ("age", "fill_missing") in actions
    assert ("plan", "fill_missing") in actions
    assert ("income", "log_transform") in actions          # long tail, not capping
    assert ("sex", "encode") in actions and ("plan", "encode") in actions
    assert ("date_of_birth", "add_age") in actions


def test_dropped_columns_get_no_other_advice(messy):
    r = dst.profile(messy, charts=False)
    for col in ("blank", "country", "notes"):
        assert [s.action for s in by_column(r, col)] == ["drop"]


def test_small_count_columns_skip_outlier_advice(messy):
    r = dst.profile(messy, charts=False)
    assert by_column(r, "visits") == []


def test_identifier_is_not_encoded(messy):
    r = dst.profile(messy, charts=False)
    assert [s.action for s in by_column(r, "id")] == [None]


def test_encoding_methods(messy):
    r = dst.profile(messy, charts=False)
    methods = {s.column: s.options["method"] for s in r.suggestions if s.action == "encode"}
    assert methods == {"sex": "label", "plan": "onehot"}


def test_suggestions_follow_cleaning_order(messy):
    steps = [s.step for s in dst.profile(messy, charts=False).suggestions]
    order = ["duplicates", "drop", "type", "missing", "outliers", "transform", "encode", "derive", "note"]
    positions = [order.index(s) for s in steps]
    assert positions == sorted(positions)


def test_generated_code_runs(messy):
    r = dst.profile(messy, charts=False)
    code = r.code(df_name="data")
    assert code.count("dst.encode_categorical(") == 1        # merged into one call
    namespace = {"data": messy.copy()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
    cleaned = namespace["data"]
    assert len(cleaned) == 200
    assert "blank" not in cleaned and "country" not in cleaned
    assert cleaned["age"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(cleaned["zip_text"])
    assert {"sex_label", "plan_Basic", "income_log"} <= set(cleaned.columns)


def test_clean_data_gets_few_suggestions():
    df = pd.DataFrame({"x": np.arange(100.0), "group": ["a", "b"] * 50})
    r = dst.profile(df, charts=False)
    assert [s.action for s in r.suggestions] == ["encode"]


def test_html_is_self_contained_and_escaped(tmp_path, messy):
    messy = messy.rename(columns={"plan": "<script>alert(1)</script>"})
    r = dst.profile(messy, title="Report <b>x</b>")
    html = r.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "Report &lt;b&gt;x&lt;/b&gt;" in html
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html   # no external resources
    path = r.save(tmp_path / "report.html")
    assert path.read_text(encoding="utf-8") == html


def test_jupyter_display_and_repr(messy):
    r = dst.profile(messy, charts=False)
    assert isinstance(r, Profile)
    assert "<style>" in r._repr_html_()
    assert "202 rows" in repr(r)


def test_correlations_flag_redundant_columns():
    x = np.arange(50.0)
    df = pd.DataFrame({"a": x, "b": x * 2 + 1, "c": np.sin(x)})
    r = dst.profile(df, charts=False)
    notes = [s.message for s in r.suggestions if s.column == "b"]
    assert any("correlated with a" in m for m in notes)
    assert r.correlations is not None and r.correlations.shape == (3, 3)


def test_rejects_unusable_input():
    with pytest.raises(ValueError):
        dst.profile(pd.DataFrame())
    with pytest.raises(ValueError, match="more than once"):
        dst.profile(pd.DataFrame([[1, 2]], columns=["a", "a"]))


def test_steps_already_applied_are_not_suggested_again(messy):
    done = dst.log_transform(messy, ["income"], suffix="_log", verbose=False)
    done = dst.encode_categorical(done, {"sex": "label", "plan": "onehot"}, verbose=False)
    actions = {(s.column, s.action) for s in dst.profile(done, charts=False).suggestions}
    assert ("income", "log_transform") not in actions
    assert ("sex", "encode") not in actions and ("plan", "encode") not in actions
