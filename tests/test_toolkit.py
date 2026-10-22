import matplotlib

matplotlib.use("Agg")  # draw off-screen so tests run without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import toolkit


@pytest.fixture(autouse=True)
def close_figures(monkeypatch):
    """Skip the on-screen show() call and close every figure after each test."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


@pytest.fixture
def grades():
    return pd.Series([72, 37, 56, 41, np.nan, 76, 37, np.nan], name="Grade")


# ---------- data helpers ----------

def test_drop_nulls_removes_missing_rows(grades):
    cleaned = toolkit.drop_nulls(grades)
    assert cleaned.isna().sum() == 0
    assert len(cleaned) == 6


def test_drop_nulls_returns_data_when_nothing_is_missing():
    # The original notebook version returned None in this case
    s = pd.Series([1, 2, 3])
    cleaned = toolkit.drop_nulls(s)
    assert cleaned is not None
    pd.testing.assert_series_equal(cleaned, s)


def test_drop_nulls_does_not_modify_input(grades):
    toolkit.drop_nulls(grades)
    assert grades.isna().sum() == 2


def test_add_passed_uses_threshold():
    df = pd.DataFrame({"Grade": [59, 60, 85]})
    result = toolkit.add_passed(df)
    assert result["Passed"].tolist() == [False, True, True]
    assert "Passed" not in df.columns  # original left unchanged


def test_add_passed_custom_threshold_and_name():
    df = pd.DataFrame({"Score": [40, 50]})
    result = toolkit.add_passed(df, grade_col="Score", threshold=50, name="OK")
    assert result["OK"].tolist() == [False, True]


# ---------- regression ----------

def test_fit_line_recovers_known_line():
    x = pd.Series([0, 1, 2, 3, 4])
    m, b, predict = toolkit.fit_line(x, 3 * x + 5)
    assert m == pytest.approx(3)
    assert b == pytest.approx(5)
    assert predict(10) == pytest.approx(35)


def test_fit_line_ignores_missing_pairs():
    x = pd.Series([0, 1, 2, np.nan, 4])
    y = pd.Series([1, 3, 5, 7, np.nan])
    m, b, _ = toolkit.fit_line(x, y)
    assert m == pytest.approx(2)
    assert b == pytest.approx(1)


def test_fit_line_matches_student_data():
    df = pd.read_csv("StudentData.csv")
    m, b, predict = toolkit.fit_line(df.StudyHours, df.Grade)
    assert m == pytest.approx(7.119, abs=0.001)
    assert b == pytest.approx(2.533, abs=0.001)
    assert predict(9) == pytest.approx(66.6, abs=0.1)


# ---------- summary output ----------

def test_show_eda_prints_statistics(grades, capsys):
    toolkit.show_eda(grades)
    out = capsys.readouterr().out
    assert "Minimum: 37.0" in out
    assert "Maximum: 76.0" in out
    assert "Mode:    37.0" in out


def test_show_eda_reports_no_mode_when_values_are_unique(capsys):
    toolkit.show_eda(pd.Series([1, 2, 3]))
    assert "none (every value appears once)" in capsys.readouterr().out


# ---------- plots: each should draw without errors, even with missing values ----------

@pytest.mark.parametrize("plot", [
    toolkit.show_eda_table,
    toolkit.show_histogram,
    toolkit.show_distribution,
    toolkit.show_density,
    toolkit.show_distribution_profile,
])
def test_plots_handle_missing_values(plot, grades):
    plot(grades)
    assert plt.get_fignums(), f"{plot.__name__} did not create a figure"


def test_distribution_profile_has_four_panels(grades):
    toolkit.show_distribution_profile(grades)
    assert len(plt.gcf().axes) == 4
