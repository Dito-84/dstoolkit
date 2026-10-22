# Data_Science_Toolkit

[![CI](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Dito-84/Data_Science_Toolkit/actions/workflows/ci.yml)

A small Python module for quick exploratory data analysis (distributions, summary tables and line fits), plus the notebooks it was built in. It works on any numeric pandas column and handles missing values on its own.

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

## Functions

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
| `Toolkit_on_Titanic_Data.ipynb` | the toolkit applied unchanged to 1,309 Titanic passengers: missing ages, heavily skewed fares, fare by class |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite checks the data helpers, the line fit (against a known line and the student data) and that every plot draws without errors when values are missing. GitHub Actions runs the tests and every notebook on each push and pull request.

## Data

- `StudentData.csv`: 17 students' study hours and grades (two rows deliberately missing)
- `titanic3.csv`: Titanic passenger list from the Vanderbilt Biostatistics dataset, with `?` for missing values
