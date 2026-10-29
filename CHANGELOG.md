# Changelog

## 0.6.0

- **`dst.validate(df, rules, checks)`**: checks data against rules you write and reports every problem, without changing the data. Rules per column: `type`, `required`, `min`/`max`, `allowed`, `unique` and `pattern`. `checks` compare two columns in each row, e.g. `("discharge_date", ">=", "admission_date")`. Returns a `ValidationReport` with `.ok`, `.summary()`, `.failures()`, `.failing_index` and `.raise_if_failed()`.
  - Nothing is guessed or converted: the text "12" is not a number, 1 is not True (also in `allowed`, `unique` and `==` checks), and date text is not read as a date.
  - Missing values are skipped by every rule except `required`, which also flags empty text (but not "N/A").
  - A column that doesn't exist is reported as a failure with the closest name. Mistakes in the rules (a misspelled rule name, a bad regular expression, `allowed` given as text, `min` above `max`) raise before any data is read. `ColumnRule` is a `TypedDict`, so type checkers catch misspelled rule names too.
  - `pattern` always uses Python's `re`, so a pattern means the same thing on pandas 2 and 3.
  - The rule format is experimental until 1.0.
- **`profile(rules=..., checks=...)`** adds a Validation section, a "Rules failed" tile and notes on each column's card. Without them the report is unchanged. Validation results are not turned into cleaning suggestions.
- `clean()` is unchanged; to validate its output, use `dst.validate(dst.clean(df), rules)`.
- The `Profile_Titanic` notebook validates the raw and cleaned data, and the example report in `docs/` includes the Validation section.

## 0.5.0

New functions, rewritten from earlier sandbox versions so they can't silently damage data:

- **Type conversion:** `to_boolean`, `to_datetime_columns`, `to_category`, and `convert_types` to run them together. Also available as an optional first step in `clean(types=True)`.
  - Yes/No conversion ignores case and only converts columns where every value is a yes/no word; anything else is left alone (or raises, when you name the column).
  - Date conversion only touches text that looks like dates. Number columns are never converted (the old version turned a "days" column of 3s and 5s into 1970 dates), and plain numbers in text such as "12" aren't read as dates.
- **Charts:** `show_counts` (sorted bars with counts and shares; replaces pie and doughnut charts), `show_grouped_box` (a number across groups, sorted by median, with group sizes) and `show_scatter` (optional bubble sizes and trend line). Each returns the figure so it can be saved.
- **`profile()`** now suggests converting yes/no text to True/False and date text to dates, and puts those steps first in `report.code()`. Yes/no text is no longer suggested for label-encoding.
- `fill_missing(categorical="missing")` now adds "Missing" to category columns, and leaves gaps in True/False columns as missing rather than failing.

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
