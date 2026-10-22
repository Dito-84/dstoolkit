"""Reusable helpers for quick exploratory data analysis.

These are the functions developed in the notebooks in this repo, collected in
one place so any notebook or script can use them:

    from toolkit import show_eda, show_distribution_profile
    show_distribution_profile(df["Grade"])

Every function drops null values first, so a column with missing data is
safe to pass in directly.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

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
]


def drop_nulls(data):
    """Return a copy of a Series or DataFrame with null rows removed.

    Always returns data (the unchanged copy when there are no nulls), so the
    result can be used safely in the next step.
    """
    return data.dropna(axis=0)


def add_passed(df, grade_col="Grade", threshold=60, name="Passed"):
    """Return a copy of df with a True/False column for grade >= threshold."""
    out = df.copy()
    out[name] = out[grade_col] >= threshold
    return out


def show_eda(var_data):
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


def show_eda_table(var_data, decimals=2):
    """Draw the .describe() statistics for a Series as a table figure."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots()
    ax.axis("off")
    pd.plotting.table(ax, data.describe().round(decimals), loc="center",
                      cellLoc="center", colWidths=[0.4, 0.4])
    plt.show()


def _mark_center(ax, data, mode=True):
    """Draw dashed lines for mean (cyan), median (red) and optionally mode (yellow)."""
    ax.axvline(x=data.mean(), color="cyan", linestyle="dashed", linewidth=2, label="Mean")
    ax.axvline(x=data.median(), color="red", linestyle="dashed", linewidth=2, label="Median")
    if mode:
        ax.axvline(x=data.mode()[0], color="yellow", linestyle="dashed", linewidth=2, label="Mode")
    ax.legend()


def show_histogram(var_data, title="Data Distribution"):
    """Histogram with mean, median and mode lines."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data)
    _mark_center(ax, data)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.show()


def show_distribution(var_data, title="Data Distribution"):
    """Histogram (with center lines) stacked above a horizontal boxplot."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    ax[0].hist(data)
    ax[0].set_ylabel("Frequency")
    _mark_center(ax[0], data)
    ax[1].boxplot(data, vert=False)
    ax[1].set_xlabel("Value")
    fig.suptitle(title)
    plt.show()


def show_density(var_data, title="Data Density"):
    """Kernel density curve with mean and median lines."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot.density(ax=ax)
    _mark_center(ax, data, mode=False)
    ax.set_title(title)
    plt.show()


def show_distribution_profile(var_data, title="Data Distribution"):
    """2x2 overview: histogram, density curve, boxplot and summary table."""
    data = drop_nulls(var_data)
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 7.2))

    ax[0, 0].hist(data)
    ax[0, 0].set_ylabel("Frequency")
    _mark_center(ax[0, 0], data, mode=False)

    data.plot.density(ax=ax[0, 1])
    data.plot.box(ax=ax[1, 0], vert=False)

    ax[1, 1].axis("off")
    pd.plotting.table(ax[1, 1], data.describe().round(2), loc="center",
                      cellLoc="center", colWidths=[0.4, 0.4])

    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.25)
    plt.show()


def fit_line(x, y):
    """Fit y = m*x + b by least squares and return (m, b, predict).

    `predict` is a function that uses the fitted m and b, so predictions
    always match the data instead of hardcoded numbers.
    """
    pairs = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1).dropna()
    result = stats.linregress(pairs["x"], pairs["y"])
    m, b = result.slope, result.intercept

    def predict(value):
        return m * value + b

    return m, b, predict
