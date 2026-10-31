# Changelog

## Unreleased

- **New: `dst.associations(df)`** shows which columns are related: one row per pair, with the test, the strength, the p-value, the adjusted p-value and the number of rows used.
  - Two number columns: Pearson or Spearman. `method="auto"` uses Pearson only when both columns look normally distributed.
  - Two category columns: chi-square, with bias-corrected Cramér's V.
  - A number and a category column: ANOVA, with the correlation ratio eta.
  - Each pair is tested once, on the rows where both values are present, and p-values are corrected for testing many pairs (Benjamini-Hochberg).
  - Notes flag chi-square tests with small expected counts and eta inflated by small groups.
  - IDs, free text, dates and constant columns are left out, and the summary names them.
  - This replaces earlier drafts that dropped missing values from each column separately, which paired values from different rows: r = 0.978 instead of 1.0 in the regression test.
- **New: `dst.standardize_labels(df, columns, mapping)`** gives each label one spelling.
  - The mapping is `{"Female": ["f", "woman"], "Male": ["m", "man"]}`, or a list of labels when spellings only differ by case and spaces.
  - Matching ignores case and extra spaces. Missing values stay missing, and category columns stay category columns.
  - Values that match nothing are kept by default (`unmatched="missing"` or `"raise"` if you prefer), and the summary always lists them. The 2023 draft this replaces silently blanked any value it didn't recognise.
- **`profile()`** suggests `standardize_labels` when a category column has labels that differ only by case or spaces, and runs it before encoding in `report.code()`. Its encoding suggestion now counts categories after standardizing.

## 0.8.0

The cleaning steps can now be learned on training data and applied to new data, so they fit into machine-learning pipelines without leaking test data. Also: optional scikit-learn, summaries through `logging`, a hardened browser app, and stricter CI.

- **New: scikit-learn transformers** in `dstoolkit.transformers`, also available as `dst.MissingFiller`, `dst.OutlierCapper`, `dst.CategoryEncoder` and `dst.Scaler`:
  - `fit()` learns the fill values, outlier bounds, categories or ranges from the training data, and `transform()` applies those same values to any data.
  - They work in a scikit-learn `Pipeline`, so cross-validation refits them on each training fold.
  - On the data they were fitted on, they give the same results as `fill_missing`, `handle_outliers(action="cap")`, `encode_categorical` and `normalize`.
  - `CategoryEncoder` handles categories not seen in training, either ignoring them or raising, with `handle_unknown=`.
  - They take and return DataFrames, never change the input, and never add or remove rows.
  - They need the `sklearn` extra. `import dstoolkit` still works without it.
- **Browser app:** the browser now enforces "your file never leaves this page" instead of relying on trust. See `app/README.md`:
  - a Content-Security-Policy limits scripts and requests to the site itself and one pinned Pyodide release, with no inline scripts and no `eval`
  - `pyodide.js` loads with an integrity hash
  - Excel support is served from the site at pinned versions. Before, it was fetched from PyPI on each Excel upload: harmless, since only the package was downloaded, but it was a server the page didn't mention.
  - The browser tests fail on any policy violation.
- **Notebooks:** `learning/Data_Cleaning_Pipeline` now explains each step, from one-off charts to the packaged functions. The code is unchanged.
- **Summaries can go to Python's `logging`.** Every one-line summary ("Filled missing values: age (263)") is now sent through the standard `logging` module, on the `"dstoolkit"` logger at INFO level. New `dst.set_message_output()`:
  - `"print"` (default) prints them exactly as before.
  - `"logging"` hands them to your application's logging setup, with its timestamps and format, wherever its logs go (a file, CloudWatch).
  - `"off"` silences them everywhere.
  `verbose=False` still silences a single call. `show_eda()` still prints, because printing the summary is what it's for. Reloading the package (`importlib.reload`, Jupyter's `%autoreload`) never makes messages print twice.
- **scikit-learn is now optional.** It's needed only for the transformers and `fill_missing(numeric="knn")`, so a plain install no longer pulls it in. Install `dstoolkit[sklearn]`, or scikit-learn itself. Without it, those raise an error that says what to install; everything else works as before.
- **Development:** a new CI job runs the tests with the oldest supported versions (pandas 2.0, NumPy 1.24, matplotlib 3.10, SciPy 1.10, scikit-learn 1.3), so the minimums in `pyproject.toml` are tested rather than assumed.
- **Development:** CI now lints with Ruff (style, import order, likely bugs, modern syntax) and measures test coverage, failing below 95% (currently about 97%). `pip install -e ".[dev]"` installs both. No change to how the package behaves.

## 0.7.0

Fixes for problems found in a full review: several functions could change data in ways you wouldn't notice. Some defaults change, so this is a minor release rather than a patch.

**Changed defaults (check your code):**

- **`clean()` fills nothing unless asked.** `missing` now defaults to off (it was on), so `clean(df)` only removes exact duplicate rows. Before, it filled every numeric column, IDs included, with its mean, and every text column with its most common value. Pass `missing=True` or `missing={...}` to fill gaps.
- **Rows keep their labels.** `remove_duplicates`, `handle_outliers(action="remove")` and `fill_missing`'s removal of empty rows no longer renumber the index, so rows can be traced back to the raw data and match `validate().failing_index`. Call `.reset_index(drop=True)` if you want 0, 1, 2, ... again.
- **`to_datetime_columns(dayfirst=None)`** is the new default: the day/month order is worked out from the data (see below). `dayfirst=True` and `dayfirst=False` work as before, except that contradicting values are no longer swapped.

**Bug fixes:**

- **Day and month could be swapped within one column.** `to_datetime_columns` read each value on its own, so in one column "01/02/2020" became 2 January while "13/02/2020" became 13 February. The order is now decided once per column: a day above 12 shows which way round the dates are written, and when every day is 12 or less the summary says month/day was assumed. A column that mixes both orders is left as text with a note (or raises, when you name it), and with `dayfirst` set, values that contradict it become missing instead of being read the other way round.
- **Year-first dates were misread with `dayfirst=True`.** pandas reads "2020-01-02" as 1 February in that mode; dates that start with the year are now always read year-month-day.
- **`fill_missing` changed date columns.** `categorical="missing"` wrote the text "Missing" into date columns, turning them into text (pandas warns this will become an error), and `"most_frequent"` filled missing dates with the most common date. Gaps in date columns now stay missing, with a note.
- **`fill_missing(numeric="knn")`**:
  - crashed when a numeric column was entirely empty; such columns are now left empty, with a note
  - measured similarity on unscaled values, so a column with large numbers (income) decided on its own which rows were neighbours; columns are now put on the same scale first
  - new `knn_features=` chooses which columns judge similarity, so IDs and prediction targets can be left out
- **Mean or median fills crashed on nullable integer columns** (`Int64`) when the value had decimals. The column is now stored as decimals (`Float64`), with a note; whole-number fills keep the integer type.
- **`add_age_column`** returned negative ages for birth dates after the reference date, and silently left unreadable dates blank. Both now give a missing age and are counted in the summary. Text dates are read with the same day/month rules as `to_datetime_columns`.
- **`profile()`** flags date columns that mix day-first and month-first order (and no longer suggests a conversion or an age column for them), and its earliest/latest dates use the same reading.
- `show_eda_table`, `show_histogram`, `show_distribution`, `show_density` and `show_distribution_profile` now return their figure, like the other chart functions, so they can be saved.

**Docs:** the README has a new "What it's for, and what it isn't" section: dstoolkit is for exploring and cleaning a dataset once, not a modelling pipeline, because its functions learn from the data they're given and have no separate fit and transform step.

**Other:** the repository is now `Dito-84/dstoolkit` (was `Dito-84/Data_Science_Toolkit`), matching the package name. Old links and `git+https://` installs redirect, but the README, package metadata and the browser app now use the new address.

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
