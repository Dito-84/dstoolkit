# Changelog

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
