"""dst.validate(): single-column rules (issue #9, stage A).

All data here is made up. Text columns are tested both as plain Python
objects and as pandas' string type, because pandas 3 stores text as strings
by default and pandas 2.3 doesn't.
"""

import re

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst

TEXT_DTYPES = ["object", "string"]


def run(df, rules=None, **kwargs):
    return dst.validate(df, rules, verbose=False, **kwargs)


def failing(report, column, rule):
    return report.failures(column=column, rule=rule)["row"].tolist()


# ---------- the example from issue #9 ----------

def test_issue_example_reports_each_problem():
    df = pd.DataFrame({
        "age": [34, 134, None, -1],
        "sex": ["male", "female", "Male", "female"],
        "patient_id": ["P0001", "P0002", "P0002", "X1"],
    })
    report = run(df, {
        "age": {"type": "number", "min": 0, "max": 120, "required": True},
        "sex": {"allowed": ["male", "female"]},
        "patient_id": {"unique": True, "pattern": r"^P\d{4}$"},
    })
    assert not report.ok
    assert failing(report, "age", "required") == [2]
    assert failing(report, "age", "min") == [3]
    assert failing(report, "age", "max") == [1]
    assert failing(report, "sex", "allowed") == [2]
    assert failing(report, "patient_id", "unique") == [1, 2]
    assert failing(report, "patient_id", "pattern") == [3]
    assert report.failing_index.tolist() == [1, 2, 3]


def test_clean_data_passes():
    df = pd.DataFrame({"age": [1, 2, 3], "sex": ["male", "female", "male"]})
    report = run(df, {"age": {"type": "integer", "min": 0, "unique": True}, "sex": {"allowed": ["male", "female"]}})
    assert report.ok
    assert report.failing_index.empty
    assert report.failures().empty
    assert list(report.failures().columns) == ["row", "column", "rule", "value", "reason"]
    assert report.raise_if_failed() is report


# ---------- the data is never changed ----------

def test_input_is_not_modified():
    df = pd.DataFrame({"a": [1, None, 300], "b": pd.Series(["x", "y", None], dtype="string"),
                       "c": pd.Categorical(["u", "v", "u"])})
    before = df.copy()
    run(df, {"a": {"type": "integer", "min": 0, "max": 10, "required": True, "unique": True},
             "b": {"pattern": "x", "allowed": ["x"], "type": "text", "required": True},
             "c": {"type": "category", "allowed": ["u"]}})
    pd.testing.assert_frame_equal(df, before)


# ---------- missing values ----------

@pytest.mark.parametrize("blank", [None, np.nan, pd.NA])
def test_missing_values_are_skipped_by_every_rule_except_required(blank):
    df = pd.DataFrame({"x": pd.Series([5, blank, blank], dtype=object)})
    report = run(df, {"x": {"type": "number", "min": 0, "max": 10, "allowed": [5], "unique": True}})
    assert report.ok
    assert set(report.summary()["checked"]) == {1}


def test_required_flags_missing_values_and_empty_text_but_not_na_words():
    df = pd.DataFrame({"name": ["Ann", None, "", "   ", "N/A"]})
    report = run(df, {"name": {"required": True}})
    out = report.failures()
    assert out["row"].tolist() == [1, 2, 3]
    assert out["reason"].tolist() == ["missing", "empty text", "empty text"]
    assert report.summary()["examples"][0].startswith("row 1 = missing; row 2 = ''")
    # "N/A" is a value someone typed; validate doesn't guess it means blank
    assert 4 not in out["row"].tolist()


def test_required_finds_missing_dates_and_nullable_numbers():
    df = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", None]),
                       "n": pd.array([1, None], dtype="Int64")})
    report = run(df, {"when": {"required": True}, "n": {"required": True}})
    assert failing(report, "when", "required") == [1]
    assert failing(report, "n", "required") == [1]


def test_unique_does_not_count_blanks_as_repeats():
    df = pd.DataFrame({"id": ["a", None, None, "b"]})
    assert run(df, {"id": {"unique": True}}).ok


# ---------- type ----------

@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_number_stored_as_text_fails_number(dtype):
    df = pd.DataFrame({"age": pd.Series(["12", "abc"], dtype=dtype)})
    out = run(df, {"age": {"type": "number"}}).failures()
    assert out["row"].tolist() == [0, 1]
    assert "pd.to_numeric" in out["reason"][0]
    assert "text, not a number" in out["reason"][1]


def test_mixed_column_reports_only_the_wrong_values():
    df = pd.DataFrame({"age": [30, "forty", 25.5, True, None]})
    report = run(df, {"age": {"type": "number"}})
    assert failing(report, "age", "type") == [1, 3]   # True is not a number


def test_true_false_column_is_not_a_number():
    df = pd.DataFrame({"flag": [True, False]})
    report = run(df, {"flag": {"type": "number"}})
    assert report.summary()["failed"].tolist() == [2]


def test_integer_accepts_whole_floats_but_not_fractions():
    # pandas stores whole numbers as floats when there are blanks
    df = pd.DataFrame({"n": [3.0, None, 3.5, np.inf]})
    assert failing(run(df, {"n": {"type": "integer"}}), "n", "type") == [2, 3]


def test_integer_on_mixed_objects():
    df = pd.DataFrame({"n": pd.Series([3, 3.0, 3.5, "3"], dtype=object)})
    assert failing(run(df, {"n": {"type": "integer"}}), "n", "type") == [2, 3]


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_text_type(dtype):
    text = pd.DataFrame({"s": pd.Series(["a", "b"], dtype=dtype)})
    assert run(text, {"s": {"type": "text"}}).ok
    mixed = pd.DataFrame({"s": ["a", 1]})
    assert failing(run(mixed, {"s": {"type": "text"}}), "s", "type") == [1]
    numbers = pd.DataFrame({"s": [1, 2]})
    assert not run(numbers, {"s": {"type": "text"}}).ok


def test_boolean_type_accepts_real_true_false_only():
    assert run(pd.DataFrame({"b": [True, False]}), {"b": {"type": "boolean"}}).ok
    assert run(pd.DataFrame({"b": pd.array([True, None], dtype="boolean")}), {"b": {"type": "boolean"}}).ok
    # 1/0 and yes/no are not True/False, matching dst.to_boolean's safe default
    df = pd.DataFrame({"b": [True, 1, 0, "yes"]})
    out = run(df, {"b": {"type": "boolean"}}).failures()
    assert out["row"].tolist() == [1, 2, 3]
    assert "dst.to_boolean" in out["reason"][0]


def test_boolean_type_on_0_1_integer_column():
    df = pd.DataFrame({"b": [0, 1, 1]})
    assert run(df, {"b": {"type": "boolean"}}).summary()["failed"].tolist() == [3]


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_datetime_type(dtype):
    dates = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    assert run(dates, {"d": {"type": "datetime"}}).ok
    text = pd.DataFrame({"d": pd.Series(["2024-01-01", "soon"], dtype=dtype)})
    out = run(text, {"d": {"type": "datetime"}}).failures()
    assert out["row"].tolist() == [0, 1]                 # date text is not converted
    assert "dst.to_datetime_columns" in out["reason"][0]


def test_category_type_checks_how_the_column_is_stored():
    cat = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    assert run(cat, {"c": {"type": "category"}}).ok
    plain = pd.DataFrame({"c": ["a", "b"]})
    report = run(plain, {"c": {"type": "category"}})
    assert not report.ok
    assert "not category" in report.summary()["examples"][0]
    assert report.failures().empty      # a whole-column problem, not row failures


def test_rules_work_on_category_columns():
    df = pd.DataFrame({"c": pd.Categorical(["a", "b", "z", None], categories=["a", "b", "z", "unused"])})
    report = run(df, {"c": {"allowed": ["a", "b"], "type": "text", "pattern": "[ab]"}})
    assert failing(report, "c", "allowed") == [2]
    assert failing(report, "c", "pattern") == [2]
    assert failing(report, "c", "type") == []


# ---------- min / max ----------

def test_range_on_numbers_including_nullable_and_infinity():
    df = pd.DataFrame({"a": [-1, 0, 120, 121], "b": pd.array([-1, None, 5, 200], dtype="Int64"),
                       "c": [np.inf, -np.inf, 1.0, np.nan]})
    report = run(df, {"a": {"min": 0, "max": 120}, "b": {"min": 0, "max": 120}, "c": {"min": 0, "max": 10}})
    assert failing(report, "a", "min") == [0]
    assert failing(report, "a", "max") == [3]
    assert failing(report, "b", "min") == [0]
    assert failing(report, "b", "max") == [3]
    assert failing(report, "c", "min") == [1]
    assert failing(report, "c", "max") == [0]


def test_range_on_mixed_column_fails_non_numbers():
    df = pd.DataFrame({"a": [5, "7", True, 50]})
    out = run(df, {"a": {"max": 10}}).failures()
    assert out["row"].tolist() == [1, 2, 3]
    assert out["reason"].tolist() == ["text, not a number", "True/False, not a number", "above the maximum 10"]


def test_range_on_dates_written_as_text_in_the_rule():
    df = pd.DataFrame({"admitted": pd.to_datetime(["2019-12-31", "2024-06-01", "2031-01-01"])})
    report = run(df, {"admitted": {"min": "2020-01-01", "max": "2030-12-31"}})
    assert failing(report, "admitted", "min") == [0]
    assert failing(report, "admitted", "max") == [2]
    assert report.summary()["expected"].tolist() == ["≥ 2020-01-01", "≤ 2030-12-31"]


def test_date_range_on_text_column_does_not_parse_the_text():
    df = pd.DataFrame({"d": ["2024-01-01"]})
    out = run(df, {"d": {"min": "2020-01-01"}}).failures()
    assert out["reason"].tolist() == ["text, not a date"]


def test_date_bound_on_number_column_is_a_rule_mistake():
    with pytest.raises(TypeError, match="is a date"):
        run(pd.DataFrame({"a": [1]}), {"a": {"min": "2020-01-01"}})
    with pytest.raises(TypeError, match="is a number"):
        run(pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])}), {"d": {"min": 0}})


def test_time_zone_mismatch_is_a_rule_mistake():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"]).tz_localize("UTC")})
    with pytest.raises(TypeError, match="time zone"):
        run(df, {"d": {"min": "2020-01-01"}})
    assert run(df, {"d": {"min": "2020-01-01 00:00+00:00"}}).ok


# ---------- allowed ----------

def test_allowed_keeps_true_apart_from_1():
    # Python says 1 == True, so a plain `in` check would wrongly pass these
    df = pd.DataFrame({"x": pd.Series([True, 1, 0, False], dtype=object)})
    assert failing(run(df, {"x": {"allowed": [True, False]}}), "x", "allowed") == [1, 2]
    df = pd.DataFrame({"x": pd.Series([1, True], dtype=object)})
    assert failing(run(df, {"x": {"allowed": [1]}}), "x", "allowed") == [1]


def test_allowed_treats_1_and_1_0_as_the_same_number():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert failing(run(df, {"x": {"allowed": [1, 2]}}), "x", "allowed") == [2]


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_allowed_is_exact_but_hints_at_case_differences(dtype):
    df = pd.DataFrame({"sex": pd.Series(["male", "Male", " female", "other"], dtype=dtype)})
    out = run(df, {"sex": {"allowed": ["male", "female"]}}).failures()
    assert out["row"].tolist() == [1, 2, 3]
    assert "'male' ignoring case" in out["reason"][0]
    assert "'female' ignoring case" in out["reason"][1]
    assert out["reason"][2] == "not one of the allowed values"


def test_allowed_on_dates():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    report = run(df, {"d": {"allowed": [pd.Timestamp("2024-01-01")]}})
    assert failing(report, "d", "allowed") == [1]


def test_long_allowed_list_is_summarised():
    report = run(pd.DataFrame({"x": [1]}), {"x": {"allowed": list(range(20))}})
    assert report.summary()["expected"][0] == "one of 20 values"


def test_unhashable_value_fails_allowed_and_unique_instead_of_crashing():
    df = pd.DataFrame({"x": pd.Series([[1], "a"], dtype=object)})
    report = run(df, {"x": {"allowed": ["a"], "unique": True}})
    assert failing(report, "x", "allowed") == [0]
    assert failing(report, "x", "unique") == [0, 1]


# ---------- unique ----------

@pytest.mark.parametrize("dtype", TEXT_DTYPES + ["int64"])
def test_unique_reports_every_copy(dtype):
    values = ["1", "2", "1", "1"] if dtype != "int64" else [1, 2, 1, 1]
    df = pd.DataFrame({"id": pd.Series(values, dtype=dtype)})
    out = run(df, {"id": {"unique": True}}).failures()
    assert out["row"].tolist() == [0, 2, 3]
    assert set(out["reason"]) == {"repeated (3 times)"}


def test_unique_keeps_true_apart_from_1():
    df = pd.DataFrame({"x": pd.Series([1, True, 1.0], dtype=object)})
    assert failing(run(df, {"x": {"unique": True}}), "x", "unique") == [0, 2]


# ---------- pattern ----------

@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_pattern_must_match_the_whole_value(dtype):
    df = pd.DataFrame({"id": pd.Series(["P0001", "P00012", "xP0001"], dtype=dtype)})
    # no ^ or $: a partial match must still fail
    assert failing(run(df, {"id": {"pattern": r"P\d{4}"}}), "id", "pattern") == [1, 2]


def test_pattern_fails_numbers_that_lost_leading_zeros():
    df = pd.DataFrame({"zip": pd.Series(["02134", 2134], dtype=object)})
    out = run(df, {"zip": {"pattern": r"\d{5}"}}).failures()
    assert out["row"].tolist() == [1]
    assert "leading zeros" in out["reason"][0]


def test_pattern_on_number_column_fails_every_value():
    df = pd.DataFrame({"zip": [2134, 33901]})
    assert run(df, {"zip": {"pattern": r"\d{5}"}}).summary()["failed"].tolist() == [2]


def test_pattern_uses_python_regex_on_every_pandas_version():
    # a backreference: supported by Python's re but not by some other regex engines
    df = pd.DataFrame({"x": pd.Series(["abab", "abcd"], dtype="string")})
    assert failing(run(df, {"x": {"pattern": r"(ab)\1"}}), "x", "pattern") == [1]


# ---------- columns ----------

def test_missing_column_is_reported_with_a_suggestion():
    df = pd.DataFrame({"age": [1]})
    report = run(df, {"agee": {"required": True}, "zzz": {"required": True}})
    assert not report.ok
    notes = report.summary()["examples"].tolist()
    assert notes[0] == "column not found; did you mean 'age'?"
    assert notes[1] == "column not found"


def test_duplicate_column_names_raise():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="more than once"):
        run(df, {"a": {"required": True}})


def test_non_text_column_names():
    df = pd.DataFrame({0: [1, -1]})
    report = run(df, {0: {"min": 0}})
    assert report.summary()["column"].tolist() == ["0"]
    assert failing(report, "0", "min") == [1]


# ---------- mistakes in the rules raise before anything is checked ----------

@pytest.mark.parametrize("rules, error, match", [
    ({"a": {"requird": True}}, ValueError, "did you mean 'required'"),
    ({"a": {"type": "float"}}, ValueError, "type must be one of"),
    ({"a": {"pattern": "("}}, ValueError, "invalid pattern"),
    ({"a": {"allowed": "male"}}, TypeError, "list of values"),
    ({"a": {"allowed": []}}, ValueError, "empty"),
    ({"a": {"allowed": [[1]]}}, TypeError, "simple values"),
    ({"a": {"min": 10, "max": 1}}, ValueError, "greater than max"),
    ({"a": {"min": 0, "max": "2020-01-01"}}, TypeError, "both be numbers or both be dates"),
    ({"a": {"min": "soon"}}, ValueError, "number or a date"),
    ({"a": {"min": None}}, TypeError, "number or a date"),
    ({"a": {"min": float("nan")}}, ValueError, "can't be missing"),
    ({"a": {"required": "yes"}}, TypeError, "True or False"),
    ({"a": {"pattern": 5}}, TypeError, "regular expression"),
    ({"a": {}}, ValueError, "No rules"),
    ({"a": True}, TypeError, "must be a dict"),
    (["a"], TypeError, "dict of"),
])
def test_rule_mistakes_raise(rules, error, match):
    with pytest.raises(error, match=re.escape(match) if "(" not in match else match):
        run(pd.DataFrame({"a": [1]}), rules)


def test_false_switches_a_rule_off():
    df = pd.DataFrame({"a": [None, None]})
    report = run(df, {"a": {"required": False, "unique": False, "type": "number"}})
    assert report.summary()["rule"].tolist() == ["type"]


def test_needs_a_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        dst.validate(pd.Series([1]), {"a": {"required": True}})  # type: ignore[arg-type]


# ---------- the report ----------

def test_rows_are_reported_by_index_label():
    df = pd.DataFrame({"age": [5, 500, 7]}, index=["ann", "bo", "cy"])
    report = run(df, {"age": {"max": 120}})
    assert report.failing_index.tolist() == ["bo"]
    assert df.loc[report.failing_index, "age"].tolist() == [500]
    assert "row 'bo' = 500" in report.summary()["examples"][0]


def test_repeated_index_labels_are_still_reported_per_row():
    df = pd.DataFrame({"age": [500, 5, 600]}, index=[1, 1, 2])
    assert run(df, {"age": {"max": 120}}).failures()["row"].tolist() == [1, 2]


def test_summary_columns_and_values():
    df = pd.DataFrame({"age": [1, 200, 300, 4]})
    s = run(df, {"age": {"max": 120}}).summary()
    assert list(s.columns) == ["column", "rule", "expected", "checked", "failed", "failed_%", "passed", "examples"]
    row = s.iloc[0]
    assert (row["checked"], row["failed"], row["failed_%"], row["passed"]) == (4, 2, 50.0, False)
    assert row["examples"] == "row 1 = 200; row 2 = 300"


def test_failures_can_be_filtered():
    df = pd.DataFrame({"a": [-1, 1], "b": [-1, -1]})
    report = run(df, {"a": {"min": 0}, "b": {"min": 0, "max": 5}})
    assert len(report.failures()) == 3
    assert len(report.failures(column="b")) == 2
    assert len(report.failures(rule="max")) == 0


def test_examples_are_capped_but_counts_are_exact():
    df = pd.DataFrame({"x": range(100)})
    report = run(df, {"x": {"max": 10}}, max_examples=2)
    s = report.summary().iloc[0]
    assert s["failed"] == 89
    assert s["examples"].count("row") == 2
    assert len(report.failures()) == 89


def test_raise_if_failed_carries_the_report():
    report = run(pd.DataFrame({"a": [-1]}), {"a": {"min": 0}})
    with pytest.raises(dst.ValidationError) as err:
        report.raise_if_failed()
    assert err.value.report is report
    assert isinstance(err.value, ValueError)
    assert "a: min (≥ 0) failed for 1 row(s)" in str(err.value)


def test_prints_one_line_plus_one_per_failed_rule(capsys):
    df = pd.DataFrame({"a": [-1, 2, -3], "b": ["x", "y", "z"]})
    dst.validate(df, {"a": {"min": 0}, "b": {"required": True}})
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Validated 3 rows against 2 rule(s): 1 failed (2 row(s))."
    assert lines[1] == "  a: min (≥ 0) failed for 2 row(s), e.g. row 0 = -1; row 2 = -3"
    assert len(lines) == 2


def test_verbose_false_is_silent(capsys):
    run(pd.DataFrame({"a": [-1]}), {"a": {"min": 0}})
    assert capsys.readouterr().out == ""


def test_numpy_values_print_as_plain_numbers():
    report = run(pd.DataFrame({"a": np.array([500], dtype=np.int64)}), {"a": {"max": 1}})
    assert report.failures()["value"][0] == 500
    assert type(report.failures()["value"][0]) is int
    assert "np." not in report.message()


def test_html_display_escapes_values():
    df = pd.DataFrame({"<b>": ["<script>"]})
    html = run(df, {"<b>": {"allowed": ["ok"]}})._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_repr():
    report = run(pd.DataFrame({"a": [1]}), {"a": {"min": 0}})
    assert repr(report) == "<ValidationReport: 1 rows, 1 rule(s), 0 failed>"


# ---------- edge sizes ----------

def test_empty_dataframe_passes_but_still_reports_missing_columns():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    report = run(df, {"a": {"required": True, "min": 0}, "b": {"required": True}})
    s = report.summary()
    assert s["checked"].tolist() == [0, 0, 0]
    assert s["passed"].tolist() == [True, True, False]


def test_large_dataframe_counts_are_exact():
    n = 1_000_000
    rng = np.random.default_rng(0)
    age = rng.integers(0, 100, n).astype(float)
    age[::1000] = 150          # 1,000 too high
    age[5::1000] = np.nan      # 1,000 blank
    df = pd.DataFrame({"age": age})
    report = run(df, {"age": {"type": "number", "min": 0, "max": 120, "required": True}})
    s = report.summary().set_index("rule")
    assert s.loc["max", "failed"] == 1000
    assert s.loc["required", "failed"] == 1000
    assert s.loc["type", "failed"] == 0
    assert s.loc["max", "examples"].count("row") == 5
    assert len(report.failing_index) == 2000


# ---------- checks: comparing two columns (stage B) ----------

@pytest.fixture
def stays():
    return pd.DataFrame({
        "admitted": pd.to_datetime(["2024-01-05", "2024-01-05", None, "2024-02-01", "2024-03-01"]),
        "discharged": pd.to_datetime(["2024-01-07", "2024-01-03", "2024-01-09", "2024-02-01", None]),
        "planned_days": [2, 3, 4, 0, 1],
        "actual_days": [2, -2, 4.0, 0, 5],
    })


def test_check_reports_rows_where_the_comparison_is_false(stays):
    report = run(stays, checks={"discharged on or after admission": ("discharged", ">=", "admitted")})
    assert not report.ok
    out = report.failures()
    assert out["row"].tolist() == [1]
    assert out["reason"][0] == "2024-01-03 >= 2024-01-05 is false"
    s = report.summary().iloc[0]
    assert (s["column"], s["rule"], s["expected"]) == ("discharged on or after admission", "compare",
                                                      "discharged >= admitted")


def test_check_skips_rows_where_either_side_is_missing(stays):
    report = run(stays, checks={"order": ("discharged", ">=", "admitted")})
    assert report.summary()["checked"][0] == 3       # rows 2 and 4 have a blank side


@pytest.mark.parametrize("op, failing_rows", [
    ("==", [1, 4]), ("!=", [0, 2, 3]), ("<", [0, 1, 2, 3]),
    ("<=", [1]), (">", [0, 2, 3, 4]), (">=", [4]),
])
def test_every_operator_on_numbers(stays, op, failing_rows):
    # planned vs actual: 2/2, 3/-2, 4/4.0, 0/0, 1/5
    report = run(stays, checks={"days": ("planned_days", op, "actual_days")})
    assert failing(report, "days", "compare") == failing_rows


def test_checks_and_rules_together(stays):
    report = run(stays, {"planned_days": {"min": 0}, "actual_days": {"min": 0}},
                 checks={"order": ("discharged", ">=", "admitted")})
    assert report.summary()["rule"].tolist() == ["min", "min", "compare"]
    assert report.failing_index.tolist() == [1]


def test_failures_can_be_filtered_by_a_column_in_a_check(stays):
    report = run(stays, checks={"order": ("discharged", ">=", "admitted")})
    assert len(report.failures(column="admitted")) == 1
    assert len(report.failures(column="order")) == 1
    assert len(report.failures(rule="compare")) == 1


def test_check_values_are_both_sides(stays):
    report = run(stays, checks={"order": ("discharged", ">=", "admitted")})
    assert report.failures()["value"][0] == (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05"))
    assert "row 1 = 2024-01-03 vs 2024-01-05" in report.summary()["examples"][0]


def test_check_equality_keeps_true_apart_from_1():
    df = pd.DataFrame({"a": pd.Series([True, 1, "x"], dtype=object), "b": pd.Series([1, 1, "x"], dtype=object)})
    report = run(df, checks={"same": ("a", "==", "b")})
    assert failing(report, "same", "compare") == [0]


def test_check_equality_on_text(stays):
    df = pd.DataFrame({"a": pd.Series(["x", "y"], dtype="string"), "b": pd.Series(["x", "z"], dtype="string")})
    assert failing(run(df, checks={"same": ("a", "==", "b")}), "same", "compare") == [1]


def test_check_on_mixed_columns_fails_rows_that_cant_be_compared():
    df = pd.DataFrame({"a": pd.Series([1, "x", 5], dtype=object), "b": [2, 2, 2]})
    out = run(df, checks={"a below b": ("a", "<", "b")}).failures()
    assert out["row"].tolist() == [1, 2]
    assert out["reason"].tolist() == ["can't compare text with a number", "5 < 2 is false"]


def test_check_on_object_dates():
    df = pd.DataFrame({"a": pd.Series([pd.Timestamp("2024-01-02"), "soon"], dtype=object),
                       "b": pd.to_datetime(["2024-01-01", "2024-01-01"])})
    out = run(df, checks={"a after b": ("a", ">", "b")}).failures()
    assert out["row"].tolist() == [1]
    assert out["reason"][0] == "can't compare text with a date"


@pytest.mark.parametrize("a, b, op, match", [
    ([1, 2], pd.to_datetime(["2024-01-01", "2024-01-02"]), "==", "can't be compared"),
    (["x", "y"], ["x", "z"], "<", "numbers or dates"),
    ([True, False], [True, True], ">", "numbers or dates"),
])
def test_check_on_columns_that_cant_be_compared_is_a_rule_mistake(a, b, op, match):
    df = pd.DataFrame({"a": pd.Series(a, dtype="string" if isinstance(a[0], str) else None), "b": b})
    with pytest.raises(TypeError, match=match):
        run(df, checks={"c": ("a", op, "b")})


def test_check_time_zone_mismatch_is_a_rule_mistake():
    df = pd.DataFrame({"a": pd.to_datetime(["2024-01-01"]).tz_localize("UTC"),
                       "b": pd.to_datetime(["2024-01-01"])})
    with pytest.raises(TypeError, match="time zone"):
        run(df, checks={"c": ("a", ">=", "b")})


def test_check_with_missing_column_is_reported(stays):
    report = run(stays, checks={"order": ("discharged", ">=", "admited")})
    assert not report.ok
    assert report.summary()["examples"][0] == "column 'admited' not found; did you mean 'admitted'?"


@pytest.mark.parametrize("checks, error, match", [
    ({"c": ("a", "=>", "b")}, ValueError, "operator must be one of"),
    ({"c": ("a", ">=")}, TypeError, "must be \\(column, operator, column\\)"),
    ({"c": ["a", ">=", "b"]}, TypeError, "must be \\(column, operator, column\\)"),
    ({"c": lambda d: d["a"] > 0}, TypeError, "functions aren't supported yet"),
    ({"": ("a", ">=", "b")}, TypeError, "needs a name"),
    (("a", ">=", "b"), TypeError, "dict of"),
])
def test_check_mistakes_raise(checks, error, match):
    with pytest.raises(error, match=match):
        run(pd.DataFrame({"a": [1], "b": [2]}), checks=checks)


def test_needs_rules_or_checks():
    with pytest.raises(ValueError, match="rules, checks or both"):
        dst.validate(pd.DataFrame({"a": [1]}))


def test_check_with_duplicate_column_names_raises():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="more than once"):
        run(df, checks={"c": ("a", "<", "b")})


def test_checks_do_not_modify_input(stays):
    before = stays.copy()
    run(stays, checks={"order": ("discharged", ">=", "admitted"), "days": ("planned_days", "==", "actual_days")})
    pd.testing.assert_frame_equal(stays, before)


def test_check_is_printed_by_name(stays, capsys):
    dst.validate(stays, checks={"discharged on or after admission": ("discharged", ">=", "admitted")})
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[1] == ("  discharged on or after admission: (discharged >= admitted) failed for 1 row(s), "
                        "e.g. row 1 = 2024-01-03 vs 2024-01-05")


def test_large_check_counts_are_exact():
    n = 1_000_000
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) % 365, unit="D")
    end = start + pd.Timedelta(days=3)
    end = end.where(np.arange(n) % 1000 != 0, start - pd.Timedelta(days=1))   # 1,000 bad rows
    df = pd.DataFrame({"start": start, "end": end})
    report = run(df, checks={"order": ("end", ">=", "start")})
    assert report.summary()["failed"][0] == 1000
