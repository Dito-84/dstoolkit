"""Type conversion: to_boolean, to_datetime_columns, to_category, convert_types.

Several tests reproduce failures found in the earlier sandbox versions of
these functions, so they can't come back.
"""

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst


# ---------- to_boolean ----------

def test_yes_no_in_any_case_becomes_true_false():
    # the sandbox version crashed on mixed case ("Yes" and "no")
    df = pd.DataFrame({"done": ["Yes", "no", "YES", " No "]})
    out = dst.to_boolean(df, verbose=False)
    assert out["done"].dtype == "boolean"
    assert out["done"].tolist() == [True, False, True, False]


def test_short_and_true_false_words_count():
    df = pd.DataFrame({"a": ["Y", "n"], "b": ["True", "false"]})
    out = dst.to_boolean(df, verbose=False)
    assert out["a"].tolist() == [True, False]
    assert out["b"].tolist() == [True, False]


def test_blank_words_become_missing():
    df = pd.DataFrame({"done": ["Yes", "N/A", None, "", "null", "No"]})
    out = dst.to_boolean(df, verbose=False)
    assert out["done"].isna().tolist() == [False, True, True, True, True, False]


def test_column_with_other_values_is_left_alone():
    # the sandbox version crashed on any column that also held another value
    df = pd.DataFrame({"answer": ["Yes", "No", "Unknown"], "name": ["Ann", "Bo", "Cy"]})
    out = dst.to_boolean(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


def test_named_column_with_other_values_raises():
    df = pd.DataFrame({"answer": ["Yes", "No", "Unknown"]})
    with pytest.raises(ValueError, match="aren't yes/no"):
        dst.to_boolean(df, ["answer"], verbose=False)


def test_zero_one_text_only_converts_when_named():
    # a text column of 0s and 1s is often a count or a code, so it's never guessed
    df = pd.DataFrame({"flag": ["1", "0", "1"]})
    assert dst.to_boolean(df, verbose=False)["flag"].tolist() == ["1", "0", "1"]
    assert dst.to_boolean(df, ["flag"], verbose=False)["flag"].tolist() == [True, False, True]


def test_to_boolean_does_not_modify_input_and_reports(capsys):
    df = pd.DataFrame({"done": ["Yes", "No"]})
    dst.to_boolean(df)
    assert df["done"].tolist() == ["Yes", "No"]
    assert "Converted to True/False: done." in capsys.readouterr().out


def test_existing_bool_and_number_columns_untouched():
    df = pd.DataFrame({"b": [True, False], "n": [1, 0]})
    pd.testing.assert_frame_equal(dst.to_boolean(df, verbose=False), df)


# ---------- to_datetime_columns ----------

def test_date_text_becomes_dates():
    df = pd.DataFrame({"admit": ["01/02/2020", "03/04/2020", None], "name": ["a", "b", "c"]})
    out = dst.to_datetime_columns(df, verbose=False)
    assert pd.api.types.is_datetime64_any_dtype(out["admit"])
    assert out["admit"].iloc[0] == pd.Timestamp("2020-01-02")
    assert out["admit"].isna().iloc[2]
    assert out["name"].tolist() == ["a", "b", "c"]


def test_number_column_named_days_is_not_turned_into_dates():
    # the sandbox version turned [3, 5] into 1970-01-01 because the name contains "day"
    df = pd.DataFrame({"Length of stay (days)": [3, 5], "Admission date": ["2020-01-01", "2020-02-01"]})
    out = dst.to_datetime_columns(df, verbose=False)
    assert out["Length of stay (days)"].tolist() == [3, 5]
    assert pd.api.types.is_datetime64_any_dtype(out["Admission date"])


def test_number_text_is_not_turned_into_dates():
    # dateutil reads "12" as the 12th of this month; plain numbers are never dates here
    df = pd.DataFrame({"count": ["12", "7", "3"], "zip": ["33615", "33602", "34102"]})
    pd.testing.assert_frame_equal(dst.to_datetime_columns(df, verbose=False), df)


def test_naming_a_number_column_raises():
    df = pd.DataFrame({"days": [3, 5]})
    with pytest.raises(TypeError, match="1970"):
        dst.to_datetime_columns(df, ["days"], verbose=False)


def test_mostly_non_dates_are_skipped_unless_named(capsys):
    df = pd.DataFrame({"mixed": ["2020-01-01", "soon", "later", "2020-03-01"]})
    assert dst.to_datetime_columns(df, verbose=False)["mixed"].tolist() == df["mixed"].tolist()
    out = dst.to_datetime_columns(df, ["mixed"])
    assert out["mixed"].isna().sum() == 2
    assert "2 value(s) in 'mixed' couldn't be read as dates" in capsys.readouterr().out


def test_dayfirst_and_format():
    df = pd.DataFrame({"d": ["01/02/2020", "13/02/2020"]})
    out = dst.to_datetime_columns(df, dayfirst=True, verbose=False)
    assert out["d"].tolist() == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-13")]
    compact = pd.DataFrame({"d": ["20200131", "20200201"]})
    out = dst.to_datetime_columns(compact, ["d"], format="%Y%m%d", verbose=False)
    assert out["d"].tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-01")]


def test_min_share_must_be_a_fraction():
    with pytest.raises(ValueError):
        dst.to_datetime_columns(pd.DataFrame({"d": ["2020-01-01"]}), min_share=0)


# ---------- to_category ----------

def test_repeated_labels_become_categories_but_unique_text_does_not():
    df = pd.DataFrame({"hospital": ["North", "South", "North", "South", "North", "East"],
                       "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"]})
    out = dst.to_category(df, verbose=False)
    assert isinstance(out["hospital"].dtype, pd.CategoricalDtype)
    assert not isinstance(out["patient_id"].dtype, pd.CategoricalDtype)


def test_max_unique_limits_auto_conversion_but_not_named_columns():
    df = pd.DataFrame({"code": list("abcd") * 5})
    assert not isinstance(dst.to_category(df, max_unique=3, verbose=False)["code"].dtype, pd.CategoricalDtype)
    assert isinstance(dst.to_category(df, ["code"], max_unique=3, verbose=False)["code"].dtype, pd.CategoricalDtype)


# ---------- convert_types, clean, fill_missing ----------

@pytest.fixture
def discharges():
    return pd.DataFrame({
        "hospital": ["North", "South", "North", "South"],
        "admit": ["2020-01-03", "2020-02-11", "2020-03-15", None],
        "complication": ["No", "Yes", "no", None],
        "los_days": [3, 5, 2, 4],
    })


def test_convert_types_converts_booleans_and_dates_only_by_default(discharges):
    out = dst.convert_types(discharges, verbose=False)
    assert out["complication"].dtype == "boolean"
    assert pd.api.types.is_datetime64_any_dtype(out["admit"])
    assert out["los_days"].tolist() == [3, 5, 2, 4]
    assert not isinstance(out["hospital"].dtype, pd.CategoricalDtype)
    out = dst.convert_types(discharges, categories=True, verbose=False)
    assert isinstance(out["hospital"].dtype, pd.CategoricalDtype)


def test_clean_runs_types_first_only_when_asked(discharges):
    untouched = dst.clean(discharges, missing=False, verbose=False)
    assert untouched["complication"].tolist()[:3] == ["No", "Yes", "no"]
    typed = dst.clean(discharges, types=True, missing=False, verbose=False)
    assert typed["complication"].dtype == "boolean"


def test_fill_missing_leaves_true_false_gaps_and_handles_categories(discharges, capsys):
    typed = dst.convert_types(discharges, categories=True, verbose=False)
    typed.loc[0, "hospital"] = np.nan
    out = dst.fill_missing(typed, categorical="missing")
    assert out["complication"].isna().sum() == 1           # stays missing, no crash
    assert out["hospital"].iloc[0] == "Missing"             # "Missing" added as a category
    assert "Left gaps in True/False column(s) complication as missing" in capsys.readouterr().out
    filled = dst.fill_missing(typed, categorical="most_frequent", verbose=False)
    assert filled["complication"].isna().sum() == 0


def test_profile_suggests_conversions_and_its_code_runs(discharges):
    report = dst.profile(discharges, charts=False)
    actions = {(s.column, s.action) for s in report.suggestions}
    assert ("complication", "to_boolean") in actions
    assert ("admit", "to_datetime") in actions
    assert ("complication", "encode") not in actions        # yes/no text isn't label-encoded any more
    code = report.code()
    if "dst.fill_missing(" in code:   # conversions come before any filling
        assert code.index("dst.to_boolean(") < code.index("dst.fill_missing(")
    scope = {"df": discharges.copy()}
    exec(code, scope)
    assert scope["df"]["complication"].dtype == "boolean"
    assert pd.api.types.is_datetime64_any_dtype(scope["df"]["admit"])


# ---------- day/month order (0.7.0) ----------

def test_day_month_order_is_worked_out_from_the_data_not_per_value():
    # 0.6.0 read each value on its own: "01/02/2020" became 2 January but
    # "13/02/2020" became 13 February in the same column
    df = pd.DataFrame({"d": ["01/02/2020", "13/02/2020", "03/04/2020"]})
    out = dst.to_datetime_columns(df, verbose=False)
    assert out["d"].tolist() == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-13"),
                                 pd.Timestamp("2020-04-03")]


def test_column_mixing_both_orders_is_left_alone_or_raises_when_named(capsys):
    df = pd.DataFrame({"d": ["13/02/2020", "02/13/2020", "01/01/2020"]})
    out = dst.to_datetime_columns(df)
    assert out["d"].tolist() == df["d"].tolist()
    assert "mixes day-first and month-first" in capsys.readouterr().out
    with pytest.raises(ValueError, match="'13/02/2020' and '02/13/2020'"):
        dst.to_datetime_columns(df, ["d"])


def test_explicit_dayfirst_never_swaps_contradicting_values(capsys):
    df = pd.DataFrame({"d": ["01/02/2020", "13/02/2020"]})
    out = dst.to_datetime_columns(df, ["d"], dayfirst=False)
    assert out["d"].iloc[0] == pd.Timestamp("2020-01-02")
    assert pd.isna(out["d"].iloc[1])            # was silently read as 13 February
    assert "1 value(s) in 'd' couldn't be read" in capsys.readouterr().out


def test_year_first_dates_are_never_read_as_year_day_month():
    # pandas reads "2020-01-02" as 1 February when dayfirst=True
    df = pd.DataFrame({"d": ["2020-01-02", "13/02/2020"]})
    out = dst.to_datetime_columns(df, ["d"], dayfirst=True, verbose=False)
    assert out["d"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-13")]


def test_ambiguous_order_is_reported(capsys):
    dst.to_datetime_columns(pd.DataFrame({"d": ["01/02/2020", "03/04/2020"]}))
    assert "as month/day/year because no day was above 12" in capsys.readouterr().out


def test_profile_flags_mixed_order_dates_without_suggesting_a_conversion():
    df = pd.DataFrame({"visit": ["13/02/2020", "02/13/2020", "01/01/2020"] * 5})
    report = dst.profile(df, charts=False)
    notes = [s for s in report.suggestions if s.column == "visit"]
    assert any("mix day-first and month-first" in s.message for s in notes)
    assert "to_datetime" not in {s.action for s in notes}
