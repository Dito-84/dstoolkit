# dstoolkit

[![CI](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml)

**Data Science Toolkit**: a small, typed Python package for the first stages of a data project. It **cleans** a pandas DataFrame (duplicates, missing values, outliers, encoding, scaling) and **explores** it (summary statistics, distribution plots, line fits). Every function returns a new DataFrame, so your raw data is never changed, and every function copes with missing values on its own.

## Install

```bash
pip install git+https://github.com/Dito-84/Data_Science_Toolkit
```

Requires Python 3.10+.

## Quick start

```python
import pandas as pd
import dstoolkit as dst

raw = pd.read_csv("data/titanic3.csv", na_values="?")

df = dst.clean(
    raw,
    missing={"numeric": "median", "columns": ["age", "fare", "embarked"]},
    outliers={"columns": ["fare"], "action": "cap"},
    encode={"columns": {"sex": "label", "embarked": "onehot"}},
)
# No duplicates found.
# Filled missing values: age (263), fare (1), embarked (2)
# Capped outliers: fare (171)
# sex labels: female=0, male=1
# Encoded sex (label), embarked (onehot).

dst.show_distribution_profile(df["age"], title="Passenger Age")
```

Each cleaning step prints a one-line summary of what it changed; pass `verbose=False` to silence it.

## Cleaning: `dstoolkit.cleaning`

| Function | What it does |
|---|---|
| `remove_duplicates(df, subset, keep)` | Drops duplicate rows, optionally judged on a subset of columns |
| `fill_missing(df, numeric, categorical, columns)` | Fills gaps: mean, median, zero or k-nearest-neighbours for numbers; most frequent value or "Missing" for text |
| `find_outliers(df, columns, method)` | True/False table of outliers by IQR or z-score |
| `handle_outliers(df, columns, method, action)` | Removes outlier rows, or caps values at the bounds and keeps the rows |
| `winsorize(df, columns, lower, upper)` | Clips columns to percentiles (default 5th–95th) |
| `log_transform(df, columns, suffix)` | log(1 + x) for long right-tailed columns like income or price |
| `encode_categorical(df, columns)` | One-hot or label encoding per column; label codes keep missing values missing |
| `normalize(df, columns)` | Min-max (0–1) or z-score scaling per column |
| `add_age_column(df, dob_column, as_of)` | Age in completed years from a birth-date column, as of today or another date column |
| `clean(df, ...)` | Runs duplicates → missing values → outliers → encoding → scaling in one call |

## Exploration: `dstoolkit.exploration`

| Function | What it does |
|---|---|
| `show_eda(series)` | Prints min, max, mean, median and mode |
| `show_eda_table(series)` | `.describe()` statistics drawn as a table |
| `show_histogram(series)` | Histogram with mean, median and mode lines |
| `show_distribution(series)` | Histogram stacked above a boxplot |
| `show_density(series)` | Density curve with mean and median lines |
| `show_distribution_profile(series)` | 2×2 overview: histogram, density, boxplot, summary table |
| `fit_line(x, y)` | Least-squares line; returns slope, intercept and a `predict` function |
| `drop_nulls(data)` | Copy of a Series/DataFrame with null rows removed |
| `add_passed(df, grade_col, threshold)` | Adds a True/False column for values at or above a threshold |

Every function is also available directly from `dstoolkit`, as in the quick start.

## Notebooks

| Notebook | Covers |
|---|---|
| [`Preprocessing_Titanic`](notebooks/Preprocessing_Titanic.ipynb) | Every cleaning step on the Titanic passenger list, then the same steps in one `clean()` call |
| [`Toolkit_on_Titanic_Data`](notebooks/Toolkit_on_Titanic_Data.ipynb) | The exploration functions on real data: missing ages, heavily skewed fares, fare by class |
| [`Data_Cleaning_Pipeline`](notebooks/Data_Cleaning_Pipeline.ipynb) | Building the distribution functions step by step |
| [`Python_for_Data_Science_Matplotlib`](notebooks/Python_for_Data_Science_Matplotlib.ipynb) | Bar, pie, histogram, box, density and scatter plots; normalization; correlation; linear regression |
| [`Using_Python_for_Data_Science_Example`](notebooks/Using_Python_for_Data_Science_Example.ipynb) | NumPy arrays, building and filtering DataFrames, handling nulls, `groupby`, sorting |

## Project layout

```
src/dstoolkit/
    __init__.py       top-level API and version
    cleaning.py       duplicates, missing values, outliers, encoding, scaling, age
    exploration.py    summaries, distribution plots, line fits
    py.typed          marks the package as typed
tests/                pytest suite
notebooks/            demos and worked examples
data/                 StudentData.csv, titanic3.csv
```

## Development

```bash
git clone https://github.com/Dito-84/Data_Science_Toolkit
cd Data_Science_Toolkit
pip install -e ".[dev]"
pytest    # 51 tests
mypy      # type check
```

The tests check every function, including that inputs are never modified and that every plot draws when values are missing. GitHub Actions runs the type check, the tests and every notebook on Python 3.10 and 3.13 for each push and pull request. See [CHANGELOG.md](CHANGELOG.md) for version history.

## Data

- `data/StudentData.csv`: 17 students' study hours and grades (two rows deliberately missing)
- `data/titanic3.csv`: Titanic passenger list from the Vanderbilt Biostatistics dataset, with `?` for missing values

## License

[MIT](LICENSE) © 2026 Rolando Rodriguez Jr.

## Credits

The cleaning steps were inspired by [AutoClean](https://github.com/elisemercury/AutoClean) by Elise Landman (MIT License), an automated preprocessing pipeline for pandas. The functions here are independent implementations.
