"""Exploratory analysis: summaries, distribution plots, comparisons and quick line fits.

    import dstoolkit as dst

    dst.show_eda(df["Grade"])
    dst.show_distribution_profile(df["Grade"], title="Grade Distribution")
    m, b, predict = dst.fit_line(df["StudyHours"], df["Grade"])

    dst.show_counts(df["hospital"])                          # how often each label appears
    dst.show_grouped_box(df, "length_of_stay", by="hospital")  # a number compared across groups
    dst.show_scatter(df["StudyHours"], df["Grade"], trend=True)

Every function drops null values first, so a column with missing data is
safe to pass in directly. Chart functions return their matplotlib figure, so
it can be saved with fig.savefig(...); in Jupyter, end the line with ; to keep
the returned figure from being shown a second time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "drop_nulls",
    "add_passed",
    "show_eda",
    "show_eda_table",
    "show_histogram",
    "show_distribution",
    "show_density",
    "show_distribution_profile",
    "fit_line",
    "show_counts",
    "show_grouped_box",
    "show_scatter",
]

PandasData = TypeVar("PandasData", pd.Series, pd.DataFrame)


def drop_nulls(data: PandasData) -> PandasData:
    """Return a copy of a Series or DataFrame with null rows removed.

    Always returns data (the unchanged copy when there are no nulls), so the
    result can be used safely in the next step.
    """
    return data.dropna(axis=0)


def add_passed(df: pd.DataFrame, grade_col: str = "Grade", threshold: float = 60,
               name: str = "Passed") -> pd.DataFrame:
    """Return a copy of df with a True/False column for values >= threshold."""
    out = df.copy()
    out[name] = out[grade_col] >= threshold
    return out


def show_eda(var_data: pd.Series) -> None:
    """Print min, max, mean, median and mode for a numeric Series."""
    data = drop_nulls(var_data)
    if data.nunique() == len(data):
        modes = "none (every value appears once)"
    else:
        modes = ", ".join(str(m) for m in data.mode())
    print(
        f"Minimum: {data.min()}\n"
        f"Maximum: {data.max()}\n"
        f"Mean:    {data.mean():.2f}\n"
        f"Median:  {data.median()}\n"
        f"Mode:    {modes}"
    )


def show_eda_table(var_data: pd.Series, decimals: int = 2) -> Figure:
    """Draw the .describe() statistics for a Series as a table figure."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots()
    ax.axis("off")
    pd.plotting.table(ax, data.describe().round(decimals), loc="center",
                      cellLoc="center", colWidths=[0.4, 0.4])
    plt.show()
    return fig


def _mark_center(ax: Axes, data: pd.Series, mode: bool = True) -> None:
    """Draw dashed lines for mean (cyan), median (red) and optionally mode (yellow)."""
    ax.axvline(x=data.mean(), color="cyan", linestyle="dashed", linewidth=2, label="Mean")
    ax.axvline(x=data.median(), color="red", linestyle="dashed", linewidth=2, label="Median")
    if mode:
        ax.axvline(x=data.mode()[0], color="yellow", linestyle="dashed", linewidth=2, label="Mode")
    ax.legend()


def _trim_density(ax: Axes, data: pd.Series) -> None:
    """Keep a density curve's x-axis near the data instead of far-off tails."""
    pad = (data.max() - data.min()) * 0.1 or 1
    ax.set_xlim(data.min() - pad, data.max() + pad)


def show_histogram(var_data: pd.Series, title: str = "Data Distribution") -> Figure:
    """Histogram with mean, median and mode lines."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data)
    _mark_center(ax, data)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.show()
    return fig


def show_distribution(var_data: pd.Series, title: str = "Data Distribution") -> Figure:
    """Histogram (with center lines) stacked above a horizontal boxplot."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    ax[0].hist(data)
    ax[0].set_ylabel("Frequency")
    _mark_center(ax[0], data)
    ax[1].boxplot(data, orientation="horizontal")
    ax[1].set_xlabel("Value")
    fig.suptitle(title)
    plt.show()
    return fig


def show_density(var_data: pd.Series, title: str = "Data Density") -> Figure:
    """Kernel density curve with mean and median lines."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot.density(ax=ax)
    _trim_density(ax, data)
    _mark_center(ax, data, mode=False)
    ax.set_title(title)
    plt.show()
    return fig


def show_distribution_profile(var_data: pd.Series, title: str = "Data Distribution") -> Figure:
    """2x2 overview: histogram, density curve, boxplot and summary table."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 7.2))

    ax[0, 0].hist(data)
    ax[0, 0].set_ylabel("Frequency")
    _mark_center(ax[0, 0], data, mode=False)

    data.plot.density(ax=ax[0, 1])
    _trim_density(ax[0, 1], data)
    ax[1, 0].boxplot(data, orientation="horizontal")

    ax[1, 1].axis("off")
    pd.plotting.table(ax[1, 1], data.describe().round(2), loc="center",
                      cellLoc="center", colWidths=[0.4, 0.4])

    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.25)
    plt.show()
    return fig


def fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float, Callable[[float], float]]:
    """Fit y = m*x + b by least squares and return (m, b, predict).

    `predict` is a function that uses the fitted m and b, so predictions
    always match the data instead of hardcoded numbers.
    """
    from scipy import stats  # imported on first use; scipy is slow to import

    pairs = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1).dropna()
    result = stats.linregress(pairs["x"], pairs["y"])
    m, b = float(result.slope), float(result.intercept)

    def predict(value: float) -> float:
        return m * value + b

    return m, b, predict


def _label(series: pd.Series, fallback: str) -> str:
    return str(series.name) if series.name is not None else fallback


def show_counts(var_data: pd.Series, top: int = 15, include_missing: bool = True,
                title: str | None = None) -> Figure:
    """Horizontal bar chart of how often each value appears, largest first.

    Each bar is labeled with its count and share. Values beyond the `top` most
    common are combined into one "Other" bar, and missing values get their own
    bar unless include_missing=False. Use this instead of a pie chart: bars are
    easy to compare at a glance, even with many categories.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    counts = var_data.astype("object").where(var_data.notna(), "(missing)") if include_missing \
        else drop_nulls(var_data).astype("object")
    counts = counts.value_counts()
    missing = counts.pop("(missing)") if "(missing)" in counts.index else None
    if len(counts) > top:
        rest = counts.iloc[top:]
        counts = pd.concat([counts.iloc[:top], pd.Series({f"Other ({len(rest)} values)": rest.sum()})])
    if missing is not None:  # always last, so gaps never look like a real category
        counts = pd.concat([counts, pd.Series({"(missing)": missing})])
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.4 * len(counts) + 1)))
    labels = [str(v) for v in counts.index][::-1]   # barh draws bottom-up; largest goes on top
    sizes = [int(n) for n in counts.to_numpy()][::-1]
    bars = ax.barh(labels, sizes)
    ax.bar_label(bars, labels=[f"{n:,} ({n / total:.0%})" for n in sizes], padding=3)
    ax.set_xlim(0, counts.max() * 1.18)
    ax.set_xlabel("Count")
    ax.set_title(title or f"{_label(var_data, 'Values')}: counts")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig


def show_grouped_box(df: pd.DataFrame, value: str, by: str, max_groups: int = 20,
                     title: str | None = None) -> Figure:
    """Boxplots of a numeric column for each group, sorted by median (highest at the top).

    Useful for questions like "how does length of stay differ by hospital?".
    Each group's label shows how many rows it has, so small groups aren't over-read.
    Only the `max_groups` largest groups are drawn.
    """
    for col in (value, by):
        if col not in df.columns:
            raise KeyError(f"Column not found: {col!r}")
    if not pd.api.types.is_numeric_dtype(df[value]) or pd.api.types.is_bool_dtype(df[value]):
        raise TypeError(f"{value!r} must be a numeric column")

    data = df[[value, by]].dropna()
    groups = {str(k): g[value] for k, g in data.groupby(by, observed=True)}
    largest = sorted(groups, key=lambda k: len(groups[k]), reverse=True)[:max_groups]
    order = sorted(largest, key=lambda k: groups[k].median())  # bottom-to-top: highest median on top

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(order) + 1.5)))
    ax.boxplot([groups[k] for k in order], orientation="horizontal",
               tick_labels=[f"{k} (n={len(groups[k]):,})" for k in order])
    ax.set_xlabel(value)
    ax.set_title(title or f"{value} by {by}")
    if len(groups) > max_groups:
        ax.text(1, -0.12, f"Showing the {max_groups} largest of {len(groups)} groups",
                transform=ax.transAxes, ha="right", fontsize=9, color="gray")
    fig.tight_layout()
    plt.show()
    return fig


def show_scatter(x: pd.Series, y: pd.Series, size: pd.Series | None = None, trend: bool = False,
                 title: str | None = None) -> Figure:
    """Scatterplot of y against x, with optional bubble sizes and a fitted trend line.

    size: a third numeric Series; bubble areas are scaled to it (a bubble chart)
    trend: add the least-squares line from fit_line() and show its equation
    Rows missing x, y or size are left out.
    """
    parts = {"x": pd.Series(x).reset_index(drop=True), "y": pd.Series(y).reset_index(drop=True)}
    if size is not None:
        parts["size"] = pd.Series(size).reset_index(drop=True)
    pairs = pd.concat(parts, axis=1).dropna()
    if pairs.empty:
        raise ValueError("No rows have values for both x and y")

    areas = None
    if size is not None:
        s = pairs["size"].astype(float)
        span = s.max() - s.min()
        areas = 20 + 380 * ((s - s.min()) / span if span else 0 * s + 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pairs["x"], pairs["y"], s=areas, alpha=0.6 if size is not None else 0.8)
    if trend and len(pairs) >= 2 and pairs["x"].nunique() > 1:
        m, b, predict = fit_line(pairs["x"], pairs["y"])
        xs = pd.Series([pairs["x"].min(), pairs["x"].max()])
        ax.plot(xs, xs.map(predict), color="red", linestyle="dashed", label=f"y = {m:.3g}x + {b:.3g}")
        ax.legend(loc="best")
    x_name, y_name = _label(pd.Series(x), "x"), _label(pd.Series(y), "y")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(True, alpha=0.3)
    subtitle = f" (bubble size: {_label(pd.Series(size), 'size')})" if size is not None else ""
    ax.set_title(title or f"{y_name} vs {x_name}{subtitle}")
    fig.tight_layout()
    plt.show()
    return fig
