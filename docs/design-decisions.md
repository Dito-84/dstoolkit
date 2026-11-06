# Design decisions

Why dstoolkit works the way it does. Each entry gives the decision, the reason, the alternatives considered, and what it costs. The [CHANGELOG](../CHANGELOG.md) records when each one arrived.

## Data safety

### 1. Functions never change their input

Every cleaning function returns a new DataFrame and keeps the original row labels.

- **Why:** in a notebook, a function that edits its argument in place silently changes the data every later cell sees, and re-running a cell gives a different answer. Keeping row labels means a cleaned row can always be traced back to the raw file.
- **Alternatives:** `inplace=True` options, as older pandas offered. pandas itself has moved away from them.
- **Cost:** one copy of the DataFrame per call. For the dataset sizes dstoolkit targets, that's cheap next to a wrong answer. The tests check it for every cleaning function.

### 2. When a guess could damage data, don't guess

dstoolkit leaves the data alone, or raises a clear error if you named the column, rather than guess. For example:

- 1/0 is read as True/False only in columns you name.
- Number columns are never turned into dates.
- Gaps in dates are never filled.
- `clean()` only removes exact duplicates unless you ask for more.

- **Why:** a wrong guess is worse than no guess, because it looks like clean data. Filling an ID column with its mean or reading a ZIP code as a number gives output that passes every later check and is still wrong.
- **Alternatives:** aggressive defaults that "just work" on tidy demo data, as many auto-cleaning tools do.
- **Cost:** more explicit arguments for the user. The profile report's suggestions name the argument, so the cost is usually a line of code, not a search.

### 3. Mixed day-first and month-first dates are never guessed

Each date column gets one day/month order, worked out from the whole column. If a column mixes both orders, it's flagged and left alone.

- **Why:** a value like `03/04/2024` holds no information about its own order. Guessing value by value silently swaps day and month for every date where both are 12 or under. The report counts the values that can only be day-first, only month-first, or either, and suggests fixing it at the source (for example, by which system wrote each row).
- **Alternatives:** per-value parsing (pandas' `dayfirst` with fallback) or a majority vote.
- **Cost:** some columns stay text until someone resolves them. That's deliberate.

## Preparing data for models

### 4. Two kinds of tools: one-off cleaning functions and transformers that fit on training data

The cleaning functions learn their numbers (means, ranges, categories, outlier bounds) from the DataFrame you pass in. The scikit-learn transformers learn them from the training rows only.

- **Why:** cleaning a whole dataset before splitting it leaks information from the test rows into training, and inflates scores. The functions are right for exploring and one-off cleaning. The transformers are right inside a `Pipeline`, where cross-validation refits them on each fold.
- **Alternatives:** one set of functions with a `fit` flag, which is easy to misuse.
- **Cost:** two ways to do similar things. A test makes sure that, on their training data, the transformers give exactly what the functions give.

### 5. Leaks are reported, not silently dropped

`profile(target=...)` flags columns that predict the outcome almost perfectly (strength 0.9 or more). It checks by value, by blank pattern, and for dates. `after_outcome=` lets you declare columns recorded after the outcome. The generated `pipeline()` leaves flagged columns out, with a comment saying why. The cleaning code (`code()`) never drops them on its own.

- **Why:** a leak can make a model look excellent and be useless in practice. On the Titanic demo, the lifeboat column lifts cross-validated accuracy from 0.785 to 0.976; on the synthetic readmission demo, whether a readmission cost was recorded lifts ROC AUC to 1.000. Statistics can't tell a leak from a genuinely strong feature, so a person decides.
- **Alternatives:** dropping every flagged column automatically, which would sometimes remove the best real feature.
- **Cost:** the user has to read the report, which is the point of it.

### 6. Validation that respects repeat patients and time

When an ID repeats (the same patient with several admissions), the report warns, and `pipeline(group=...)` uses grouped folds. `pipeline(time_column=...)` trains on the past and tests on the future instead.

- **Why:** with random folds, the same patient lands in both training and test data, and the model is scored partly on people it has already seen. Time-based splits answer the real question: how well does a model trained on last year do this year?
- **Alternatives:** shuffled k-fold everywhere (scikit-learn's default habit).
- **Cost:** slightly lower, more honest scores.

### 7. A cautious starter model, always compared with a no-skill baseline

The generated `pipeline()` uses histogram gradient boosting with gentle settings (learning rate 0.03, depth 3, at least 40 rows per leaf). It scores a "no-skill" baseline on the same folds.

- **Why:** this model handles blanks and needs no scaling, so it runs on messy data. scikit-learn's default settings overfit small tables: on the synthetic readmission demo they were overconfident, predicting 63% for the riskiest tenth when 48% were actually readmitted. The gentler settings improved both calibration (Brier score 0.145 → 0.135) and ranking (ROC AUC 0.71 → 0.75). The baseline makes any score meaningful: 0.75 means little until you know guessing gets 0.50.
- **Alternatives:** model selection or tuning built into the toolkit. That's out of scope. dstoolkit hands off to scikit-learn.
- **Cost:** not the best possible model, by design. It's a starting point.

## Checking data and new data

### 8. Validation reports problems and never fixes them

`validate()` checks rules you write (types, required, ranges, allowed values, uniqueness, patterns, formats, cross-column comparisons). It lists every failing row and changes nothing. It only raises when a rule itself is wrong, never because the data is bad.

- **Why:** a validator that converts values hides the problem it exists to find. The text `"12"` isn't a number, and `1` isn't True, until someone decides they are.
- **Alternatives:** coercing types first (as some schema libraries do), or raising on the first failure.
- **Cost:** you see more failures. `report.raise_if_failed()` is there for pipelines that should stop.

### 9. Drift is judged by size of change, not by p-values

`compare(reference, current)` rates each column by the population stability index (PSI: moderate from 0.1, major from 0.25), by change in blanks, and by the share of new values. P-values are reported (adjusted for many tests) but don't set the status.

- **Why:** with thousands of rows, a trivial shift is "statistically significant"; with a few dozen, a serious one isn't. Effect sizes answer the operational question: has this changed enough to matter? Rare categories (under 1% of the reference) share one bin, so a long tail of diagnosis codes doesn't look like drift. Dates (which always move forward) and identifiers are left out.
- **Alternatives:** KS or chi-square tests as the verdict.
- **Cost:** the PSI thresholds are an industry rule of thumb, not a law. They're documented so users can disagree.

### 10. Testing many column pairs means correcting for it

`associations()` tests every pair of columns, picks the test by column type and sample size, and adjusts p-values with Benjamini-Hochberg.

- **Why:** test 40 pairs at p < 0.05 and about two will look related by chance. Each pair is tested on the rows where both values exist; an earlier draft misaligned rows and reported a correlation of 0.978 for two columns that were perfectly related.
- **Alternatives:** raw p-values, or Bonferroni (too strict for exploration).
- **Cost:** fewer "significant" findings, and the ones left are more likely to be real.

## Engineering

### 11. Suggestions come as code that's tested to run

The profile report doesn't just describe problems; `code()` and `pipeline()` write the Python that fixes them, in the right order.

- **Why:** a suggestion you have to translate into code gets skipped or done wrongly. The tests run the generated code on real and synthetic data, so a suggestion that doesn't run is a failing test.
- **Cost:** the code generator has to track the order of steps (for example, dropping notes rows first and renaming columns last), which is more work than writing advice.

### 12. A browser version where the file never leaves the page

The browser app runs Python inside the page with Pyodide. A Content-Security-Policy lets the page talk only to its own site and one pinned Pyodide release, which has an integrity hash. Excel support is served from the site.

- **Why:** people won't drop a sensitive export into a website on a promise. Here the browser enforces it, and CI fails on any policy violation.
- **Alternatives:** a server-side app (the data would leave the computer), or self-hosting Pyodide (25–40 MB per deploy; worth it only if a compliance rule requires it).
- **Cost:** slower first load, and features that need SciPy (related columns) are off in the browser.

### 13. Light dependencies, tested at their oldest supported versions

scikit-learn is optional, and `import dstoolkit` never needs it. CI runs the tests on Python 3.10 (pandas 2.3), on Python 3.13 (pandas 3), and with the oldest versions the package says it supports.

- **Why:** a lower bound nobody tests is a guess. Testing the minimums keeps the promise in `pyproject.toml` true.
- **Cost:** some newer pandas features can't be used until the minimum moves.

### 14. Still 0.x on purpose

The version stays below 1.0 while the API is still changing (the validation rule format is marked experimental).

- **Why:** a 1.0 is a promise that existing code keeps working until 2.0. That promise should come after the API has settled and has outside users, not before.
