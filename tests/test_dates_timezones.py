"""Dates from real exports: ISO timestamps, UTC offsets, daylight saving and repeated row labels.

Every case is a small made-up frame whose right answer is known.
"""

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst

NY = "America/New_York"


# ---------- ISO 8601 timestamps with "T" ----------

def test_iso_timestamps_with_t_are_found_without_naming_the_column():
    # The text is lower-cased before matching, so the "T" used to arrive as "t" and miss the pattern
    df = pd.DataFrame({"t": ["2024-01-31T10:00:00", "2024-02-01T11:30:00", "2024-02-02T09:15:00"]})
    out = dst.to_datetime_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(out["t"])
    assert out["t"].tolist() == [pd.Timestamp("2024-01-31 10:00"), pd.Timestamp("2024-02-01 11:30"),
                                 pd.Timestamp("2024-02-02 09:15")]


def test_profile_suggests_converting_iso_timestamps():
    df = pd.DataFrame({"t": [f"2024-01-{d:02d}T08:00:00" for d in range(1, 31)], "x": range(30)})
    report = dst.profile(df, charts=False, related=False)
    assert [c.kind for c in report.columns if c.name == "t"] == ["datetime"]
    assert any(s.column == "t" and s.action == "to_datetime" for s in report.suggestions)


# ---------- UTC offsets ----------

def across_spring_forward() -> pd.DataFrame:
    # 10:00 in New York on each side of the 10 March 2024 change: -05:00, then -04:00
    return pd.DataFrame({"t": ["2024-03-09 10:00:00-05:00", "2024-03-11 10:00:00-04:00",
                               "2024-03-12 10:00:00-04:00"]})


def test_different_offsets_are_read_in_utc(capsys):
    out = dst.to_datetime_columns(across_spring_forward())
    assert str(out["t"].dt.tz) == "UTC"
    assert out["t"].tolist() == [pd.Timestamp("2024-03-09 15:00", tz="UTC"), pd.Timestamp("2024-03-11 14:00", tz="UTC"),
                                 pd.Timestamp("2024-03-12 14:00", tz="UTC")]
    assert "in UTC" in capsys.readouterr().out


def test_different_offsets_named_explicitly_are_read_in_utc():
    out = dst.to_datetime_columns(across_spring_forward(), columns=["t"])
    assert out["t"].dt.tz_convert(NY).dt.hour.tolist() == [10, 10, 10]   # back to local time, nothing shifted


def test_one_offset_keeps_it():
    df = pd.DataFrame({"t": ["2024-01-09 10:00:00-05:00", "2024-01-11 10:00:00-05:00"]})
    out = dst.to_datetime_columns(df)
    assert out["t"].iloc[0] == pd.Timestamp("2024-01-09 15:00", tz="UTC")
    assert out["t"].dt.hour.tolist() == [10, 10]


def with_and_without_offset() -> pd.DataFrame:
    return pd.DataFrame({"t": ["2024-03-09 10:00:00-05:00", "2024-03-11 10:00:00", "2024-03-12 10:00:00-04:00"]})


def test_some_with_and_some_without_an_offset_is_left_alone(capsys):
    out = dst.to_datetime_columns(with_and_without_offset())
    assert out["t"].tolist() == with_and_without_offset()["t"].tolist()
    assert "with and without a UTC offset" in capsys.readouterr().out


def test_some_with_and_some_without_an_offset_raises_when_named():
    with pytest.raises(ValueError, match="with and without a UTC offset"):
        dst.to_datetime_columns(with_and_without_offset(), columns=["t"])
    with pytest.raises(ValueError, match="with and without a UTC offset"):
        dst.add_days_between(with_and_without_offset().assign(u="2024-04-01"), "t", "u")


def test_convert_types_and_clean_handle_different_offsets():
    # pandas 3 raised "Mixed timezones detected" here; pandas 2 left Python objects behind
    assert str(dst.convert_types(across_spring_forward())["t"].dt.tz) == "UTC"
    assert str(dst.clean(across_spring_forward(), types=True)["t"].dt.tz) == "UTC"


def test_profile_reads_different_offsets_and_flags_missing_ones():
    rows = across_spring_forward()["t"].tolist() * 10
    report = dst.profile(pd.DataFrame({"t": rows, "x": range(30)}), charts=False, related=False)
    t = next(c for c in report.columns if c.name == "t")
    assert t.stats["earliest"] == pd.Timestamp("2024-03-09 15:00", tz="UTC")

    rows = with_and_without_offset()["t"].tolist() * 10
    report = dst.profile(pd.DataFrame({"t": rows, "x": range(30)}), charts=False, related=False)
    advice = [s for s in report.suggestions if s.column == "t"]
    assert [(s.step, s.severity, s.action) for s in advice] == [("type", "high", None)]
    assert "UTC offset" in advice[0].message


# ---------- daylight saving and calendar days ----------

@pytest.mark.parametrize("start, end, days", [
    ("2024-03-09 12:00", "2024-03-11 12:00", 2),   # spring forward: 47 hours, 2 calendar days
    ("2024-11-02 12:00", "2024-11-04 12:00", 2),   # fall back: 49 hours
    ("2024-03-10 00:30", "2024-03-10 23:30", 0),   # the 23-hour day itself
    ("2024-03-09 23:00", "2024-03-10 04:00", 1),   # past midnight across the change
])
def test_days_between_counts_calendar_days_across_daylight_saving(start, end, days):
    df = pd.DataFrame({"a": pd.to_datetime([start]).tz_localize(NY), "b": pd.to_datetime([end]).tz_localize(NY)})
    assert dst.add_days_between(df, "a", "b")["days_a_to_b"].tolist() == [days]


def test_days_between_text_with_offsets_counts_local_days():
    df = pd.DataFrame({"a": ["2024-03-09 12:00:00-05:00"], "b": ["2024-03-11 12:00:00-04:00"]})
    # read in UTC (17:00 and 16:00), still 2 calendar days apart
    assert dst.add_days_between(df, "a", "b")["days_a_to_b"].tolist() == [2]


# ---------- repeated row labels (pd.concat without ignore_index) ----------

def test_dates_with_repeated_row_labels():
    df = pd.DataFrame({"d": ["2020-01-05", "03/14/2020", "2020-02-03"]}, index=[0, 0, 1])
    out = dst.to_datetime_columns(df, columns=["d"])
    assert list(out.index) == [0, 0, 1]
    assert out["d"].tolist() == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-03-14"), pd.Timestamp("2020-02-03")]


def test_conflicting_date_orders_example_with_repeated_labels():
    df = pd.DataFrame({"d": ["13/01/2020", "01/13/2020"]}, index=[5, 5])
    with pytest.raises(ValueError, match="'13/01/2020' and '01/13/2020'"):
        dst.to_datetime_columns(df, columns=["d"])


def test_standardize_labels_with_repeated_row_labels():
    df = pd.DataFrame({"sex": ["m", "zzz", "f", None]}, index=[0, 0, 1, 1])
    out = dst.standardize_labels(df, "sex", {"Male": ["m"], "Female": ["f"]})
    assert list(out.index) == [0, 0, 1, 1]
    assert out["sex"].tolist()[:3] == ["Male", "zzz", "Female"] and pd.isna(out["sex"].iloc[3])


def test_days_between_and_age_with_repeated_row_labels():
    df = pd.DataFrame({"a": ["2024-01-01", "01/15/2024"], "b": ["2024-01-03", "2024-01-16"]}, index=[7, 7])
    assert dst.add_days_between(df, "a", "b")["days_a_to_b"].tolist() == [2, 1]
    assert dst.add_age_column(df, "a", as_of="2030-06-01")["age"].tolist() == [6, 6]


def test_associations_with_repeated_row_labels():
    # crosstab aligned the pair by label and raised; the pair is now built by position
    df = pd.DataFrame({"a": ["x", "y", None, "x", "y", "x"] * 5, "b": ["x", "y", "x", "x", None, "y"] * 5},
                      index=[0, 0, 1, 1, 2, 2] * 5)
    result = dst.associations(df, significant_only=False, verbose=False)
    assert result["n"].tolist() == [20]        # the 20 rows where both values are present
    assert dst.profile(df, charts=False).associations is not None


def test_profile_drops_notes_rows_by_position():
    # The notes row's label (0) repeats the first data row's; dropping by label removed both
    n = 25
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": ["x"] * n, "c": ["y"] * n})
    df = pd.concat([df, pd.DataFrame({"a": [np.nan], "b": ["Source: made-up data"], "c": [None]})])
    report = dst.profile(df, charts=False, related=False)
    a = next(c for c in report.columns if c.name == "a")
    assert a.count == n and a.kind == "numeric"
