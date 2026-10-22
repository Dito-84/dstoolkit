# Data_Science_Toolkit

[![CI](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml)

Two small Python modules for the first stages of a data project: **`preprocessing.py`** cleans a dataset (duplicates, missing values, outliers, encoding, scaling) and **`toolkit.py`** explores it (distributions, summary tables, line fits). Both work on any pandas DataFrame and handle missing values on their own.

## Install

```bash
pip install git+https://github.com/Dito-84/Data_Science_Toolkit
```

Or clone the repo and run `pip install -e ".[notebooks]"` to work with the notebooks too.

## Quick start

```python
import pandas as pd
import toolkit

df = pd.read_csv("StudentData.csv")
toolkit.show_eda(df.Grade)
toolkit.show_distribution_profile(df.Grade, title="Grade Distribution")
m, b, predict = toolkit.fit_line(df.StudyHours, df.Grade)
predict(9)  # ≈ 66.6
```

## Cleaning: `preprocessing.py`

Each function returns a new DataFrame (the input is never changed) and prints a one-line summary of what it did; pass `verbose=False` to silence it.

```python
import preprocessing as pp

df = pp.clean(
    raw,
    missing={"numeric": "median", "columns": ["age", "fare", "embarked"]},
    outliers={"columns": ["fare"], "action": "cap"},
    encode={"columns": {"sex": "label", "embarked": "onehot"}},
    scale={"columns": {"age": "minmax", "fare": "zscore"}},
)
```

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
| `clean(df, ...)` | Runs duplicates → missing → outliers → encoding → scaling in one call |

## Exploration: `toolkit.py`

| Function | What it does |
|---|---|
| `drop_nulls(data)` | Copy of a Series/DataFrame with null rows removed |
| `add_passed(df, grade_col, threshold)` | Adds a True/False column for values at or above a threshold |
| `show_eda(series)` | Prints min, max, mean, median and mode |
| `show_eda_table(series)` | `.describe()` statistics drawn as a table |
| `show_histogram(series)` | Histogram with mean, median and mode lines |
| `show_distribution(series)` | Histogram stacked above a boxplot |
| `show_density(series)` | Density curve with mean and median lines |
| `show_distribution_profile(series)` | 2×2 overview: histogram, density, boxplot, summary table |
| `fit_line(x, y)` | Least-squares line; returns slope, intercept and a `predict` function |

## Notebooks

| Notebook | Covers |
|---|---|
| `Using_Python_for_Data_Science_Example.ipynb` | numpy arrays, building and filtering DataFrames, handling nulls, `groupby`, sorting |
| `Python_for_Data_Science_Matplotlib.ipynb` | bar, pie, histogram, box, density and scatter plots; normalization; correlation; linear regression |
| `Data_Cleaning_Pipeline.ipynb` | building the distribution functions step by step, ending with the `toolkit.py` versions |
| `Preprocessing_Titanic.ipynb` | every cleaning step applied to the Titanic passenger list, then the same steps in one `clean()` call |
| `Toolkit_on_Titanic_Data.ipynb` | the toolkit applied unchanged to 1,309 Titanic passengers: missing ages, heavily skewed fares, fare by class |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite (46 tests) checks every cleaning function, including that inputs are never modified, plus the line fit and that every plot draws without errors when values are missing. GitHub Actions runs the tests and every notebook on each push and pull request.

## Data

- `StudentData.csv`: 17 students' study hours and grades (two rows deliberately missing)
- `titanic3.csv`: Titanic passenger list from the Vanderbilt Biostatistics dataset, with `?` for missing values

## Credits

The cleaning steps were inspired by [AutoClean](https://github.com/elisemercury/AutoClean) by Elise Landman (MIT License), an automated preprocessing pipeline for pandas. The functions here are independent implementations.
