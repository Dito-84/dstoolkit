# Data_Science_Toolkit

Notebooks and a small helper module for exploratory data analysis with pandas, matplotlib and scipy, built around a sample dataset of student study hours and grades (`StudentData.csv`).

## toolkit.py

Reusable versions of the functions developed in the notebooks. Every function drops null values first, so columns with missing data can be passed straight in.

```python
import pandas as pd
import toolkit

df = pd.read_csv("StudentData.csv")
toolkit.show_eda(df.Grade)
toolkit.show_distribution_profile(df.Grade, title="Grade Distribution")
m, b, predict = toolkit.fit_line(df.StudyHours, df.Grade)
predict(9)  # ≈ 66.6
```

| Function | What it does |
|---|---|
| `drop_nulls(data)` | Copy of a Series/DataFrame with null rows removed |
| `add_passed(df, grade_col, threshold)` | Adds a True/False `Passed` column |
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

## Run it

```bash
pip install -r requirements.txt
jupyter notebook
```
