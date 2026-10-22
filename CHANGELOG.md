# Changelog

## 0.4.1

Fixes found while connecting dstoolkit to the other portfolio projects:

- **`profile(missing_ok=[...])`**: name columns where a blank means something (e.g. "no complication", "not readmitted"). They're still reported, marked "(expected)", but get no advice to fill or drop them. Without it, a column like `complication` that's 97% blank was suggested for dropping, even though it holds the outcome being measured.
- True/False columns no longer get "label-encode" suggestions (they're already 0/1).
- ID-like digit columns (names such as `provider_id`, `zip`, `code`, or values with leading zeros) are no longer suggested for conversion to numbers, which would strip leading zeros.
- **Bug fix:** the suggested code for "numbers stored as text" now removes thousands separators before converting. Previously `"1,234.50"` became blank.

## 0.4.0

- **`dst.profile(df)`**: a one-call dataset report covering the overview (rows, duplicates, missing cells, memory), every column (type, missing values, statistics, outliers, chart) and correlations. It also suggests cleaning steps, ranked by severity, with ready-to-run `dstoolkit` code from `report.code()`. The report shows inline in Jupyter or saves as one self-contained HTML file.
- Suggestions avoid common bad advice: small count columns don't get outlier warnings; long right tails get a log transform instead of capping; columns to be dropped get no other steps; and steps already applied (an existing `_log` or encoded column) aren't suggested again.
- `Profile_Titanic` notebook and an example report in `docs/`.
- Tutorial notebooks moved to `notebooks/learning/`, so the worked examples lead.
- MIT license.

## 0.3.0

Restructured into an installable, typed package.

- The code now lives in `src/dstoolkit/`, with two modules: `cleaning` (was `preprocessing.py`) and `exploration` (was `toolkit.py`).
- Every function is available from the top level: `import dstoolkit as dst`, then `dst.clean(...)` or `dst.show_eda(...)`.
- The install name is now `dstoolkit` (`data-science-toolkit` is taken on PyPI by an unrelated project).
- Type hints on every public function, with a `py.typed` marker so editors and type checkers can use them. CI runs `mypy`.
- Cleaning functions now raise a clear error when a column name appears more than once, instead of failing inside pandas.
- Data files moved to `data/` and notebooks to `notebooks/`.
- CI tests on Python 3.10 and 3.13.

**Upgrading from 0.2:** replace `import preprocessing as pp` and `import toolkit` with `import dstoolkit as dst`. Function names and arguments are unchanged.

## 0.2.0

- Added the cleaning module (`preprocessing.py`): `remove_duplicates`, `fill_missing`, `find_outliers`, `handle_outliers`, `winsorize`, `log_transform`, `encode_categorical`, `normalize`, `add_age_column` and the `clean` pipeline.
- Added the `Preprocessing_Titanic` demo notebook.

## 0.1.0

- First packaged release of the exploration helpers (`toolkit.py`): summary statistics, distribution plots and `fit_line`.
- Tests, CI and the `Toolkit_on_Titanic_Data` demo notebook.
