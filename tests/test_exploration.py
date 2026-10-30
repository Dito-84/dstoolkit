import matplotlib

matplotlib.use("Agg")  # draw off-screen so tests run without a display

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import dstoolkit.exploration as toolkit


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
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "StudentData.csv")
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


# ---------- show_counts, show_grouped_box, show_scatter ----------

def _bar_labels(fig):
    ax = fig.axes[0]
    return [t.get_text() for t in ax.get_yticklabels()]


def test_show_counts_sorts_and_labels_missing():
    s = pd.Series(["a", "b", "a", None, "a", "b", "c"], name="grade")
    fig = toolkit.show_counts(s)
    assert _bar_labels(fig) == ["(missing)", "c", "b", "a"]   # drawn bottom-up: largest at the top
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "3 (43%)" in texts
    fig = toolkit.show_counts(s, include_missing=False)
    assert "(missing)" not in _bar_labels(fig)


def test_show_counts_groups_the_rest_as_other():
    s = pd.Series(list("aaaabbbcc") + list("defg"), name="x")
    fig = toolkit.show_counts(s, top=2)
    assert _bar_labels(fig)[0] == "Other (5 values)"
    s_missing = pd.Series(list("aaaab") + [None] * 6, name="x")   # most common, but still drawn last
    assert _bar_labels(toolkit.show_counts(s_missing))[0] == "(missing)"


def test_show_grouped_box_orders_by_median_and_shows_group_sizes():
    df = pd.DataFrame({"los": [1, 2, 3, 10, 11, 12, 5, 6], "hospital": list("AAABBBCC")})
    fig = toolkit.show_grouped_box(df, "los", by="hospital")
    assert _bar_labels(fig) == ["A (n=3)", "C (n=2)", "B (n=3)"]   # highest median at the top


def test_show_grouped_box_limits_groups_and_checks_types():
    df = pd.DataFrame({"v": range(30), "g": [f"g{i % 10}" for i in range(30)]})
    fig = toolkit.show_grouped_box(df, "v", by="g", max_groups=4)
    assert len(_bar_labels(fig)) == 4
    assert any("4 largest of 10" in t.get_text() for t in fig.axes[0].texts)
    with pytest.raises(TypeError):
        toolkit.show_grouped_box(df, "g", by="v")
    with pytest.raises(KeyError):
        toolkit.show_grouped_box(df, "missing", by="g")


def test_show_scatter_bubbles_trend_and_missing_rows():
    x = pd.Series([1, 2, 3, 4, np.nan], name="hours")
    y = pd.Series([2, 4, 6, 8, 10], name="score")
    size = pd.Series([1, 2, 3, 4, 5], name="weight")
    fig = toolkit.show_scatter(x, y, size=size, trend=True)
    ax = fig.axes[0]
    assert len(ax.collections[0].get_offsets()) == 4             # the row missing x is left out
    assert ax.get_legend().get_texts()[0].get_text() == "y = 2x + 0"
    assert "bubble size: weight" in ax.get_title()


def test_show_scatter_with_no_complete_rows_raises():
    with pytest.raises(ValueError):
        toolkit.show_scatter(pd.Series([np.nan]), pd.Series([1.0]))


@pytest.mark.parametrize("name", ["show_eda_table", "show_histogram", "show_distribution",
                                  "show_density", "show_distribution_profile"])
def test_every_chart_returns_its_figure_so_it_can_be_saved(name, grades, tmp_path):
    # before 0.7.0 these returned None, so their charts couldn't be saved or tested
    fig = getattr(toolkit, name)(grades)
    assert isinstance(fig, Figure)
    fig.savefig(tmp_path / f"{name}.png")
    assert (tmp_path / f"{name}.png").stat().st_size > 0
