# Changelog

## 0.15.1

- **`import dstoolkit` no longer loads matplotlib.** The plotting functions and the charts in `profile()` reports import it the first time they draw, so nothing changes in a notebook. `compare()`, `profile(charts=False)`, `validate()` and the cleaning functions now run where matplotlib isn't installed, such as an AWS Lambda zip package, which has a 250 MB limit that pandas, NumPy, SciPy and matplotlib together exceed. Importing is also faster. matplotlib is still a required dependency; a test checks that it stays unloaded until something is drawn.
- **Docs:** a Design decisions section in the README and [docs/design-decisions.md](docs/design-decisions.md), with the reasons, alternatives and costs behind the main choices. The README now links the live browser app.

## 0.15.0

Two problems from real hospital exports, found by reviewing the earlier 2023 drafts.

- **New: `clean_column_names(df, case="snake")`.** It removes hidden spaces and line breaks from column names, then gives every name one style: `"snake"` (`patient_age_years`), `"pascal"` (`PatientAgeYears`) or `"strip"` (spaces only, wording kept). Export headers such as `"Number of THAs Performed \n(0, 1, or 2) "` break `df["..."]` lookups when typed without their hidden characters. Names that would clash get `_2`, `_3` and are never duplicated silently; a column whose name is already clean keeps it; names that would become empty get `column_<position>`; every rename is reported.
- **`profile()` flags messy column names** (spaces at either end, line breaks, double spaces) and suggests `clean_column_names()`. `code()` runs it last, so the steps before it can still use the original names.
- **`profile()` finds notes rows at the end of an export:** trailing rows with at most a fifth of the columns filled, when the rest of the file has almost none like that (at most 50 rows, and 10% of the file; files of 20 rows or more).
  - The column analysis leaves them out, because a "Source: ..." line in an age column makes the column text and adds fake blanks.
  - The first suggestion drops them by row label, and both `code()` and `pipeline()` do that before anything else.
  - Row and missing-cell counts in the overview still describe the file as loaded.

## 0.14.0

- **New: `dst.compare(reference, current)`** checks whether new data still looks like the data a model was built on, one row per column, biggest changes first:
  - **type changes**, such as numbers arriving as text
  - **distribution shift**, as the population stability index (PSI) over the reference's deciles for numbers and over the categories for the rest. Categories under 1% of the reference share one bin, so a long tail of rare codes doesn't look like drift. Moderate from 0.1, major from 0.25.
  - **more or fewer blanks**: moderate from 2 percentage points, major from 10
  - **new values**, such as a new code or a relabelled category: moderate from 2% of new rows, major from 5%. Rarer new values are listed without raising the status.
  - **added and removed columns**

  A Kolmogorov-Smirnov or chi-square p-value, adjusted for testing many columns, is reported but doesn't set the status, because with thousands of rows tiny harmless shifts are "significant". Dates aren't compared, since they're expected to move forward. Identifiers (many values, mostly distinct) only get the type and blank checks. Nothing is changed in either DataFrame.
- **`Hospital_Readmissions`, section 8:** the 2024 admissions against the 2023 ones (all stable), then a batch with a lab unit change, "ER" instead of "Emergency", a new diagnosis code and missing prior-admission counts. All four are flagged as major.

## 0.13.1

- **`pipeline()`'s model no longer overfits.** The generated script now uses gradient boosting with small, slowly learning trees (`learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=40`) instead of scikit-learn's defaults. With the defaults, the hospital demo's risks were overconfident: the riskiest tenth of admissions were predicted 63% and 48% were readmitted, and the safest tenth 1% against 5%. With the new settings each band is within about 2 points, the Brier score falls from 0.145 to 0.135, and ROC AUC rises from 0.71 to 0.75. On the Titanic, ROC AUC is 0.859 (was 0.856).
- **`Hospital_Readmissions` checks whether the risks can be trusted:** out-of-fold risks by band against the actual readmission rate, the Brier score against always guessing the overall rate, what calling the riskiest 10% would catch (53% of them readmitted, 27% of all readmissions), and the default settings for comparison.

## 0.13.0

Dates, done safely: the remaining date limits of 0.12.0.

- **Dates are checked for leaks.** With `profile(target=...)`, each readable date column (as a date) and the days between each pair of dates are tested against the target, and shown in the Target section as date checks. A strength of 0.9 or more flags the date, or for an interval the later of the two dates, as a possible leak, and `pipeline()` leaves it out. Predicting length of stay, `discharged` minus `admitted` is the answer itself; until now the report couldn't see that, because dates weren't tested.
- **Dates of birth become ages.** `DateFeatures(ages={"age_at_admission": ("date_of_birth", "admitted")})` gives the age in completed years on another date; a birth date after that date gives a missing age, not a negative one. `pipeline()` uses it for a column named like a date of birth, measured on `time_column` or the first other date, instead of splitting the birth date into a birth month and weekday. With no other date, the birth date is left out with a comment showing how to add an age before the split.
- **New: `pipeline(time_column="admitted")`** sorts the rows by that date and uses `TimeSeriesSplit`, so each fold trains on earlier rows and tests on later ones: the honest estimate for a model that will predict future cases. With a group column, a returning patient can then appear in both, as happens in practice, and the script says so.
- **No year feature by default in `pipeline()`.** The generated `DateFeatures` step uses month and day of week only: a model that learns from the year is judged on years it has seen, and new data will come from years it hasn't. (`DateFeatures` itself still defaults to year, month and day of week.)
- **Mixed day/month dates: clearer, still not guessed.** The suggestion now counts how many values can only be day-first, how many only month-first, and how many could be either, and suggests resolving them at the source (for example, by which site or system wrote each row). No rule can read the ambiguous ones safely, so dstoolkit still doesn't try.
- The `Hospital_Readmissions` notebook compares grouped folds with a time split.

## 0.12.0

Preparing real-world data for a model: repeated patients, dates, rare codes, informative blanks and leaks you have to declare. Every feature appears in a new demo on made-up hospital data.

- **New notebook: `Hospital_Readmissions`**, on `data/synthetic_admissions.csv`: 4,090 admissions for 1,500 patients, all generated by `data/make_admissions.py` from a fixed seed (no real patients). It predicts 30-day readmission and shows each feature below on the problem it solves: grouped folds for repeated patients, a leak found automatically (`readmission_cost`, blank unless readmitted), a leak declared with `after_outcome=` (a follow-up call 30 days later), a flag for lab tests that weren't ordered, rare diagnosis codes grouped into "Other", the length of stay from two dates, and saving and reloading the fitted model.
- **New: `profile(..., target=..., after_outcome=[...])`** marks columns you know were recorded after the outcome, such as a discharge disposition or a follow-up code. They're treated as leaks even when the statistics miss them, and `pipeline()` leaves them out under their own comment. On the Titanic, `body` only exists for people who died, but its blanks show only V = 0.25.
- **New: "was missing" flags.** `MissingFiller(add_indicator=True)` or `add_indicator=[columns]` adds a 0/1 `<column>_missing` column before filling, so a blank that carries signal isn't lost. `report.pipeline(missing_flags=[...])` adds the flags you choose. They're never added by default, and flagging a possible leak raises an error.
- **New: repeated identifiers.** An ID column whose values repeat gets a warning: with several rows per patient or customer, a random split puts the same one in training and test data, and scores look better than they'll be on new people. The column must be named like an ID or be mostly unique text, and columns suggested for dropping are skipped. **`profile(..., group="patient_id")`** makes `pipeline()` split by that column (`StratifiedGroupKFold` for classes, `GroupKFold` otherwise) and leave it out of the features. In the tests, patients told apart only by a meaningless score reach ROC AUC above 0.7 with random folds, and fall below 0.6 with grouped folds.
- **New: rare-category grouping.** `CategoryEncoder(min_frequency=...)` takes a number of training rows, or a share such as 0.01, and groups categories rarer than that into one "Other" category (`other_label=`). Categories that weren't in the training data join it too. Diagnosis codes, providers and ZIP codes often have hundreds of mostly rare values, and one column for each adds little but noise. For text columns with more than 10 categories, `pipeline()` encodes them with `min_frequency=0.01`; in the tests, a column with 128 codes becomes fewer than 20 columns instead of 128.
- **New: `DateFeatures` transformer.** Splits date columns into year, month and day of week (also day, day of year, quarter or hour if asked). `intervals={"stay_days": ("admitted", "discharged")}` adds the days between two dates, and the date columns are dropped by default. It learns nothing from the data, and missing dates give missing features. **`pipeline()` no longer drops date columns:** text dates are converted before the split, and `DateFeatures` turns them into features. With two or more dates, the script comments that a duration often matters more. Only columns that mix day-first and month-first order are still left out. In the tests, adding the length of stay raises ROC AUC by more than 0.1 when the stay drives the outcome.
- **New: `profile(df, sample=20_000)`** profiles a fixed random subset of a large dataset, so reruns match. The overview's row, duplicate and missing counts and the validation still use every row, and the report says what was sampled. The README has timings: about 50 seconds for a million rows with 16 columns, or about 5 seconds with `sample=20_000`.
- A test saves a fitted `pipeline()` model with `joblib`, loads it back and checks that the predictions match.

## 0.11.1

- **`report.pipeline()` scores a no-skill baseline** on the same folds (`DummyClassifier` or `DummyRegressor`), and prints it under the model's score. A score means little without it: on the Titanic, 79% accuracy sounds good until you know that always predicting "died" gets 62%.
- **Clearer advice on swapping the model.** The script's comment said "swap in any scikit-learn model", but a model that can't handle blanks, such as logistic regression, would fail on the first blank. It now says to fill every column and add `dst.Scaler` first.

## 0.11.0

Preparing a dataset for a model: the report now knows what the model will predict.

- **New: `profile(df, target="column")`** adds a Target section:
  - **Task and class balance.** The task is classification for text, True/False, or whole-number targets with at most 10 values, and regression otherwise; `task=` overrides it. An imbalance warning appears when the smallest class is under 10% of rows.
  - **Each feature's relation to the target**, from `associations()`, tested on the rows where both are present.
  - **Possible leaks:** features almost perfectly related to the target (strength 0.9 or more), by value or by whether they're blank. On the Titanic this flags `boat`. Leaks are reported, never dropped by the cleaning code.
  - **Blanks related to the target** (strength 0.1 or more), with the line of code that keeps them as a flag column if they were known before the outcome.
  - The target itself gets no fill, encoding or outlier advice. Rows without a target get a suggestion to drop them, and `code()` writes `df.dropna(subset=[target])`.
  - `report.target` holds it all as a `TargetProfile`.
- **New: `report.pipeline()`** writes a leak-free modelling starter. Steps that learn nothing from the data (duplicates, type fixes, log transforms) run before the split. Steps that learn from it (`MissingFiller`, `OutlierCapper`, `CategoryEncoder`) go inside a scikit-learn `Pipeline`, cross-validated with shuffled folds (stratified for classes) and a suitable metric. Possible leaks, identifiers, dates and dropped columns are left out of the features, with a comment saying why. Flags for informative blanks are written as comments, because only you know whether a blank was known before the outcome.
- **New: `associations(df, target=...)`** tests only the pairs that include that column, with the p-value correction over just those pairs.
- **Fixed: bad encoding advice.** For text columns with more than 10 categories, `profile()` suggested label codes. Numbering unordered categories invents an order that linear and distance-based models take literally. It now suggests one-hot encoding, grouping rare categories first, and says label codes suit tree-based models only. The `encode_categorical()` warning for many one-hot columns says the same.
- The `Profile_Titanic` notebook has a "Prepare for a model" section, including a leak the statistics can't catch (`body`). The example report in `docs/` includes the Target section.

## 0.10.0

- **New: a "Related columns" section in `profile()`.** It lists the pairs of columns of any type that are related, from `associations()` with its defaults: Pearson or Spearman for two number columns, chi-square with Cramér's V for two category columns, and ANOVA with eta for one of each. Until now the report only showed Pearson correlations between number columns, so a relation between, say, sex and survival didn't appear.
  - The pairs are also available as `report.associations`.
  - The HTML lists the strongest 20 pairs, with each pair's notes (small expected counts, small groups).
  - `related=False` leaves the section out. Above 30 columns it's skipped, because the number of pairs grows with the square of the columns; the report says so and shows the `associations()` call to run instead.
  - The existing correlation heatmap and the "almost perfectly correlated" suggestion are unchanged.
  - The browser app leaves the section out (`related=False`): it needs SciPy, which the page doesn't load.
- The `Profile_Titanic` notebook reads the new section, including why `boat` looks only moderately related to survival when it's tested on the rows where both values are present.
- **New notebook: `Modelling_Titanic`** (added after 0.9.0). Explores which columns are related to survival with `associations()`, which also flags a column recorded after the sinking that would leak the answer; checks the model's inputs with `validate()`; then cross-validates a scikit-learn pipeline of dstoolkit transformers (shuffled 5-fold) against a no-skill baseline: about 79% accuracy against 62%, and 98% if the leaked column is added. Documentation only; no code changes.

## 0.9.0

Three features rebuilt from earlier 2023 drafts, rewritten from scratch to fix their flaws and follow dstoolkit's safe-by-default rules.

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
- **New `format` rule in `validate()`:** `"us_zip"`, `"us_phone"` or `"email"`. It reports problems and never changes the data, and each failure says what's wrong:
  - a 9-digit ZIP without its hyphen
  - a ZIP stored as a number that lost its leading zero (2134 → 02134)
  - a phone number with 7 digits, or an area code starting with 0 or 1
  - an email with no @

  It replaces the earlier ZIP and phone validator drafts, several of which replaced anything they didn't recognise with blanks.
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
