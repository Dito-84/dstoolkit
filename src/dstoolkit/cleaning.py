"""Data cleaning: duplicates, missing values, outliers, encoding and scaling.

Each function takes a DataFrame and returns a new one; the input is never
modified. Steps can be used on their own or chained with `clean()`:

    import dstoolkit as dst

    df = dst.remove_duplicates(df)
    df = dst.fill_missing(df, numeric="median")
    df = dst.handle_outliers(df, columns=["fare"], method="iqr", action="cap")
    df = dst.encode_categorical(df, {"sex": "label", "embarked": "onehot"})
    df = dst.normalize(df, {"age": "minmax", "fare": "zscore"})

    # or all at once
    df = dst.clean(df, missing={"numeric": "median"}, outliers={"columns": ["fare"], "action": "cap"})

Columns read in as text can be given proper types first:

    df = dst.convert_types(df)                  # yes/no -> True/False, date text -> dates
    df = dst.to_category(df, ["hospital"])      # repeated labels -> pandas category

Pass `verbose=False` to any function to silence its summary message.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Union

import numpy as np
import pandas as pd

NumericFill = Literal["mean", "median", "knn", "zero"]
CategoricalFill = Literal["most_frequent", "missing"]
OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["remove", "cap"]
Encoding = Literal["onehot", "label"]
Scaling = Literal["minmax", "zscore"]
StepOptions = Union[bool, Mapping[str, Any], None]

__all__ = [
    "to_boolean",
    "to_datetime_columns",
    "to_category",
    "convert_types",
    "remove_duplicates",
    "fill_missing",
    "find_outliers",
    "handle_outliers",
    "winsorize",
    "log_transform",
    "encode_categorical",
    "normalize",
    "add_age_column",
    "clean",
]

# Dates written as text: 2020-01-31, 2020/1/31, 01/31/2020, 31-1-20, optionally with a time
DATE_PATTERN = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})([ T]\d{1,2}:\d{2}.*)?\s*$")

# Words read as True / False. 1/0 and t/f only count when you name the column,
# because a text column of 0s and 1s is often a count or a code, not a yes/no answer.
TRUE_WORDS = frozenset({"yes", "y", "true"})
FALSE_WORDS = frozenset({"no", "n", "false"})
TRUE_EXPLICIT = TRUE_WORDS | {"1", "t"}
FALSE_EXPLICIT = FALSE_WORDS | {"0", "f"}
BLANK_WORDS = frozenset({"", "n/a", "na", "nan", "none", "null"})

DOB_NAME_HINTS = ("dob", "date_of_birth", "dateofbirth", "birth_date", "birthdate", "birthday")


def _say(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _check_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    repeated = [c for c in columns if (df.columns == c).sum() > 1]
    if repeated:
        raise ValueError(f"Column name(s) {repeated} appear more than once; rename duplicates first")


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    """Requested columns (default: every numeric column), checked to be numeric."""
    if columns is None:
        return list(df.select_dtypes(include="number").columns)
    columns = list(columns)
    _check_columns(df, columns)
    not_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        raise TypeError(f"Columns must be numeric: {not_numeric}")
    return columns


def _position(df: pd.DataFrame, column: str) -> int:
    """Index of a column; new columns are inserted right after it."""
    loc = df.columns.get_loc(column)
    if not isinstance(loc, int):
        raise ValueError(f"Column name {column!r} appears more than once; rename duplicates first")
    return loc


# ---------- types ----------

def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _normalized(series: pd.Series) -> pd.Series:
    """Lower-case, trimmed text, with blank-like words (N/A, null, ...) turned into missing."""
    text = series.astype("string").str.strip().str.lower()
    return text.mask(text.isin(BLANK_WORDS))


def _columns_to_check(df: pd.DataFrame, columns: Sequence[str] | None) -> tuple[list[str], bool]:
    """(columns, explicit): the named columns, or every text column when none are named."""
    if columns is None:
        return [c for c in df.columns if _is_text(df[c])], False
    columns = list(columns)
    _check_columns(df, columns)
    return columns, True


def _is_yes_no(series: pd.Series) -> bool:
    """True when every non-blank value is a yes/no word (yes/no, y/n, true/false, any case)."""
    if not _is_text(series):
        return False
    present = _normalized(series).dropna()
    return bool(len(present)) and present.isin(TRUE_WORDS | FALSE_WORDS).all()


def to_boolean(df: pd.DataFrame, columns: Sequence[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Convert yes/no columns to True/False (pandas' nullable boolean type).

    Matching ignores case and spaces: yes/no, y/n and true/false. Blank cells and
    words like "N/A" or "null" become missing values.

    columns: the columns to convert. Default: every text column whose values are
             ALL yes/no words; any other column is left exactly as it was. When you
             name columns, 1/0 and t/f also count, and a column with any other value
             raises an error instead of being guessed at.
    """
    targets, explicit = _columns_to_check(df, columns)
    true_words, false_words = (TRUE_EXPLICIT, FALSE_EXPLICIT) if explicit else (TRUE_WORDS, FALSE_WORDS)

    result = df.copy()
    converted = []
    for col in targets:
        if pd.api.types.is_bool_dtype(result[col]):
            continue
        values = _normalized(result[col])
        present = values.dropna()
        if present.empty:
            continue
        unknown = sorted(set(present) - true_words - false_words)
        if unknown:
            if explicit:
                raise ValueError(f"{col!r} has values that aren't yes/no: {unknown[:5]}")
            continue
        result[col] = values.map(lambda v: v in true_words if isinstance(v, str) else pd.NA).astype("boolean")
        converted.append(col)
    _say(verbose, f"Converted to True/False: {', '.join(map(str, converted))}." if converted
         else "No yes/no columns to convert.")
    return result


def to_datetime_columns(df: pd.DataFrame, columns: Sequence[str] | None = None, min_share: float = 0.95,
                        dayfirst: bool = False, format: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Convert columns of dates stored as text into real dates.

    columns: the columns to convert. Default: text columns where at least
             `min_share` of the values look like dates (2020-01-31, 01/31/2020,
             ...) and parse as dates. Number columns are never converted:
             turning 3 into a date gives 1970-01-01, which is never what you want.
    dayfirst: read 01/02/2020 as 1 February instead of 2 January
    format: an exact format such as "%Y%m%d", for dates the default reading misses

    Values that can't be read as dates become missing; the summary says how many.
    """
    if not 0 < min_share <= 1:
        raise ValueError("min_share must be between 0 and 1")
    targets, explicit = _columns_to_check(df, columns)

    result = df.copy()
    converted, notes = [], []
    for col in targets:
        series = result[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            if explicit:
                raise TypeError(f"{col!r} holds numbers, not dates; converting would give 1970-01-01. "
                                f"If they are dates written as numbers (e.g. 20200131), convert the column to "
                                f"text first and pass format='%Y%m%d'.")
            continue
        text = _normalized(series)
        present = text.dropna()
        if present.empty:
            continue
        if not explicit and present.str.match(DATE_PATTERN).mean() < min_share:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst,
                                    format=format if format else "mixed")
        failed = int(parsed[text.notna()].isna().sum())
        if not explicit and 1 - failed / len(present) < min_share:
            continue
        result[col] = parsed
        converted.append(col)
        if failed:
            notes.append(f"{failed} value(s) in {col!r} couldn't be read as dates and are now missing")
    message = (f"Converted to dates: {', '.join(map(str, converted))}." if converted
               else "No date columns to convert.")
    _say(verbose, " ".join([message] + [n + "." for n in notes]))
    return result


def to_category(df: pd.DataFrame, columns: Sequence[str] | None = None, max_unique: int = 50,
                verbose: bool = True) -> pd.DataFrame:
    """Store repeated text labels (hospital, region, status) as pandas' category type.

    Categories use far less memory than text, sort in a fixed order and show up
    as groups in charts and summaries.

    columns: the columns to convert. Default: text columns with at most
             `max_unique` distinct values, where values repeat (fewer distinct
             values than half the rows).
    """
    targets, explicit = _columns_to_check(df, columns)
    result = df.copy()
    converted = []
    for col in targets:
        if isinstance(result[col].dtype, pd.CategoricalDtype):
            continue
        present = result[col].dropna()
        if not explicit and (present.empty or present.nunique() > max_unique
                             or present.nunique() > len(present) / 2):
            continue
        result[col] = result[col].astype("category")
        converted.append(col)
    _say(verbose, f"Converted to category: {', '.join(map(str, converted))}." if converted
         else "No columns to convert to category.")
    return result


def convert_types(df: pd.DataFrame, booleans: bool = True, dates: bool = True, categories: bool = False,
                  verbose: bool = True) -> pd.DataFrame:
    """Give text columns their proper types, using the safe defaults of each step.

    Order: yes/no -> True/False, then date text -> dates, then (optional)
    repeated labels -> category. Categories are off by default because later
    steps that add new values, like fill_missing(categorical="missing"), need
    the "Missing" label to exist as a category first; this function's output
    works with every other dstoolkit step either way.
    """
    result = df
    if booleans:
        result = to_boolean(result, verbose=verbose)
    if dates:
        result = to_datetime_columns(result, verbose=verbose)
    if categories:
        result = to_category(result, verbose=verbose)
    return result


# ---------- duplicates ----------

def remove_duplicates(df: pd.DataFrame, subset: Sequence[str] | None = None,
                      keep: Literal["first", "last"] = "first", verbose: bool = True) -> pd.DataFrame:
    """Drop duplicate rows.

    subset: columns that define a duplicate (default: all columns)
    keep: which copy to keep, 'first' or 'last'
    """
    if subset is not None:
        _check_columns(df, subset)
    result = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    removed = len(df) - len(result)
    _say(verbose, f"Removed {removed} duplicate row(s)." if removed else "No duplicates found.")
    return result


# ---------- missing values ----------

NUMERIC_FILL = ("mean", "median", "knn", "zero", None)
CATEGORICAL_FILL = ("most_frequent", "missing", None)


def fill_missing(df: pd.DataFrame, numeric: NumericFill | None = "mean",
                 categorical: CategoricalFill | None = "most_frequent",
                 columns: Sequence[str] | None = None, n_neighbors: int = 5,
                 drop_empty_rows: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Fill missing values, with separate methods for numeric and text columns.

    columns: only fill these columns (default: all). Useful for leaving mostly
             empty columns alone rather than inventing values for them.

    numeric: 'mean', 'median', 'knn' (estimate from the n_neighbors most similar
             rows), 'zero', or None to leave numeric gaps alone
    categorical: 'most_frequent', 'missing' (fill with the word "Missing"),
                 or None to leave them alone. With 'missing', gaps in True/False
                 columns (from to_boolean) stay missing, since they can't hold text.
    drop_empty_rows: first drop rows where every value is missing
    """
    if numeric not in NUMERIC_FILL:
        raise ValueError(f"numeric must be one of {NUMERIC_FILL}, got {numeric!r}")
    if categorical not in CATEGORICAL_FILL:
        raise ValueError(f"categorical must be one of {CATEGORICAL_FILL}, got {categorical!r}")

    result = df.copy()
    if drop_empty_rows:
        before = len(result)
        result = result.dropna(how="all").reset_index(drop=True)
        if len(result) < before:
            _say(verbose, f"Dropped {before - len(result)} completely empty row(s).")

    if columns is not None:
        _check_columns(result, columns)
    targets = list(result.columns) if columns is None else list(columns)
    numeric_all = set(result.select_dtypes(include="number").columns)
    filled = result.isna().sum()
    num_cols = [c for c in targets if c in numeric_all and result[c].isna().any()]
    cat_cols = [c for c in targets if c not in numeric_all and result[c].isna().any()]

    if num_cols and numeric is not None:
        if numeric == "knn":
            try:
                from sklearn.impute import KNNImputer
            except ImportError as err:  # pragma: no cover
                raise ImportError("numeric='knn' needs scikit-learn: pip install scikit-learn") from err
            # Similarity uses every numeric column; only the target columns are filled
            all_num = list(result.select_dtypes(include="number").columns)
            imputed = pd.DataFrame(KNNImputer(n_neighbors=n_neighbors).fit_transform(result[all_num]),
                                   columns=all_num, index=result.index)
            for col in num_cols:
                result[col] = result[col].fillna(imputed[col])
        else:
            for col in num_cols:
                value = {"mean": result[col].mean(), "median": result[col].median(), "zero": 0}[numeric]
                result[col] = result[col].fillna(value)

    kept_bool = []
    if cat_cols and categorical is not None:
        for col in cat_cols:
            if categorical == "missing" and isinstance(result[col].dtype, pd.BooleanDtype):
                kept_bool.append(col)  # True/False columns can't hold the word "Missing"; the gaps stay missing
                continue
            if categorical == "most_frequent":
                mode = result[col].mode(dropna=True)
                if mode.empty:
                    continue  # column is entirely empty; nothing to copy from
                result[col] = result[col].fillna(mode.iloc[0])
            else:
                if isinstance(result[col].dtype, pd.CategoricalDtype) and "Missing" not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories("Missing")
                result[col] = result[col].fillna("Missing")

    done = {c: int(filled[c] - result[c].isna().sum()) for c in num_cols + cat_cols}
    done = {c: n for c, n in done.items() if n}
    _say(verbose, "Filled missing values: " + ", ".join(f"{c} ({n})" for c, n in done.items())
         if done else "No missing values filled.")
    if kept_bool:
        _say(verbose, f"Left gaps in True/False column(s) {', '.join(map(str, kept_bool))} as missing; "
                      f"use categorical='most_frequent' to fill them instead.")
    return result


# ---------- outliers ----------

def _bounds(series: pd.Series, method: OutlierMethod, threshold: float | None) -> tuple[float, float]:
    values = series.dropna()
    if method == "iqr":
        threshold = 1.5 if threshold is None else threshold
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        return q1 - threshold * spread, q3 + threshold * spread
    if method == "zscore":
        threshold = 3.0 if threshold is None else threshold
        mean, std = values.mean(), values.std(ddof=0)
        return mean - threshold * std, mean + threshold * std
    raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}")


def find_outliers(df: pd.DataFrame, columns: Sequence[str] | None = None, method: OutlierMethod = "iqr",
                  threshold: float | None = None) -> pd.DataFrame:
    """True/False table marking outlying values (missing values are never outliers).

    method: 'iqr' (outside threshold x interquartile range, default 1.5) or
            'zscore' (more than threshold standard deviations from the mean, default 3)
    """
    columns = _numeric_columns(df, columns)
    flags = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        low, high = _bounds(df[col], method, threshold)
        flags[col] = (df[col] < low) | (df[col] > high)
    return flags


def handle_outliers(df: pd.DataFrame, columns: Sequence[str] | None = None, method: OutlierMethod = "iqr",
                    threshold: float | None = None, action: OutlierAction = "remove",
                    verbose: bool = True) -> pd.DataFrame:
    """Remove or cap outliers in numeric columns.

    action: 'remove' drops any row with an outlier in the chosen columns;
            'cap' pulls outlying values back to the nearest bound and keeps the row.
    See find_outliers() for method and threshold.
    """
    if action not in ("remove", "cap"):
        raise ValueError(f"action must be 'remove' or 'cap', got {action!r}")
    columns = _numeric_columns(df, columns)
    flags = find_outliers(df, columns, method, threshold)

    if action == "remove":
        keep = ~flags.any(axis=1)
        result = df.loc[keep].reset_index(drop=True)
        _say(verbose, f"Removed {int((~keep).sum())} row(s) with outliers in {columns}.")
        return result

    result = df.copy()
    for col in columns:
        low, high = _bounds(df[col], method, threshold)
        result[col] = df[col].clip(lower=low, upper=high)
    counts = {c: int(flags[c].sum()) for c in columns if flags[c].any()}
    _say(verbose, "Capped outliers: " + ", ".join(f"{c} ({n})" for c, n in counts.items())
         if counts else "No outliers found.")
    return result


def winsorize(df: pd.DataFrame, columns: Sequence[str] | None = None, lower: float = 0.05,
              upper: float = 0.95, verbose: bool = True) -> pd.DataFrame:
    """Clip each column to its lower and upper percentiles (default 5th and 95th)."""
    if not 0 <= lower < upper <= 1:
        raise ValueError("Need 0 <= lower < upper <= 1")
    columns = _numeric_columns(df, columns)
    result = df.copy()
    for col in columns:
        low, high = df[col].quantile([lower, upper])
        result[col] = df[col].clip(lower=low, upper=high)
    _say(verbose, f"Winsorized {columns} at the {lower:.0%}/{upper:.0%} percentiles.")
    return result


def log_transform(df: pd.DataFrame, columns: Sequence[str], suffix: str = "",
                  verbose: bool = True) -> pd.DataFrame:
    """Apply log(1 + x) to shrink long right tails, such as income or price.

    Values must be greater than -1. With a suffix (e.g. '_log') the result goes
    in a new column and the original is kept.
    """
    columns = _numeric_columns(df, columns)
    result = df.copy()
    for col in columns:
        if (df[col].dropna() <= -1).any():
            raise ValueError(f"Column {col!r} has values <= -1; log(1 + x) is undefined there")
        result[col + suffix] = np.log1p(df[col])
    _say(verbose, f"Log-transformed {columns}.")
    return result


# ---------- encoding ----------

def encode_categorical(df: pd.DataFrame, columns: Mapping[str, Encoding] | Sequence[str],
                       onehot_limit: int = 10, drop_original: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """Turn text categories into numbers.

    columns: {column: 'onehot' or 'label'}, or a list of columns to one-hot encode
      - 'onehot' adds one 0/1 column per category (e.g. sex_female, sex_male)
      - 'label' adds a column of integer codes (column_label), in alphabetical
        order of the categories; missing values stay missing
    onehot_limit: warn when one-hot encoding would add more columns than this
    drop_original: remove the original text columns after encoding
    """
    if not isinstance(columns, Mapping):
        columns = {c: "onehot" for c in columns}
    _check_columns(df, columns)
    bad = {c: m for c, m in columns.items() if m not in ("onehot", "label")}
    if bad:
        raise ValueError(f"Encoding must be 'onehot' or 'label': {bad}")

    result = df.copy()
    for col, method in columns.items():
        if method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            if dummies.shape[1] > onehot_limit:
                warnings.warn(f"One-hot encoding {col!r} adds {dummies.shape[1]} columns; "
                              f"consider 'label' encoding instead.", stacklevel=2)
            at = _position(result, col) + 1
            for offset, name in enumerate(dummies.columns):
                result.insert(at + offset, name, dummies[name])
        else:
            categories = sorted(df[col].dropna().unique(), key=str)
            codes = {cat: i for i, cat in enumerate(categories)}
            result.insert(_position(result, col) + 1, f"{col}_label", df[col].map(codes).astype("Int64"))
            _say(verbose, f"{col} labels: " + ", ".join(f"{k}={v}" for k, v in codes.items()))
    if drop_original:
        result = result.drop(columns=list(columns))
    _say(verbose, "Encoded " + ", ".join(f"{c} ({m})" for c, m in columns.items()) + ".")
    return result


# ---------- normalization ----------

def normalize(df: pd.DataFrame, columns: Mapping[str, Scaling] | Sequence[str],
              verbose: bool = True) -> pd.DataFrame:
    """Rescale numeric columns.

    columns: {column: 'minmax' or 'zscore'}, or a list of columns to min-max scale
      - 'minmax' rescales to the 0–1 range
      - 'zscore' centres on 0 with a standard deviation of 1
    Missing values stay missing. A column with only one distinct value becomes 0.
    """
    if not isinstance(columns, Mapping):
        columns = {c: "minmax" for c in columns}
    _numeric_columns(df, list(columns))
    bad = {c: m for c, m in columns.items() if m not in ("minmax", "zscore")}
    if bad:
        raise ValueError(f"Normalization must be 'minmax' or 'zscore': {bad}")

    result = df.copy()
    for col, method in columns.items():
        values = df[col].astype(float)
        if method == "minmax":
            spread = values.max() - values.min()
            result[col] = (values - values.min()) / spread if spread else values * 0
        else:
            std = values.std(ddof=0)
            result[col] = (values - values.mean()) / std if std else values * 0
    _say(verbose, "Normalized " + ", ".join(f"{c} ({m})" for c, m in columns.items()) + ".")
    return result


# ---------- derived columns ----------

def add_age_column(df: pd.DataFrame, dob_column: str | None = None,
                   as_of: str | pd.Timestamp | None = None, name: str = "age",
                   verbose: bool = True) -> pd.DataFrame:
    """Add age in completed years, calculated from a date-of-birth column.

    dob_column: column holding birth dates; if omitted, the first column whose
                name looks like a birth date (dob, date_of_birth, birthday, ...)
    as_of: date to measure age at (default: today). Can also be the name of a
           date column, e.g. an admission date, to get age at that event.
    """
    if dob_column is None:
        candidates = [c for c in df.columns
                      if any(h in str(c).lower().replace(" ", "_") for h in DOB_NAME_HINTS)]
        if not candidates:
            raise KeyError("No date-of-birth column found; pass dob_column=")
        dob_column = candidates[0]
    _check_columns(df, [dob_column])

    dob = pd.to_datetime(df[dob_column], errors="coerce")
    if isinstance(as_of, str) and as_of in df.columns:
        ref = pd.to_datetime(df[as_of], errors="coerce")
    else:
        ref = pd.Series(pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize(),
                        index=df.index)

    before_birthday = (ref.dt.month < dob.dt.month) | ((ref.dt.month == dob.dt.month) & (ref.dt.day < dob.dt.day))
    age = (ref.dt.year - dob.dt.year - before_birthday.astype(int)).astype("Int64")

    result = df.copy()
    if name in result.columns:
        result[name] = age
        _say(verbose, f"Updated {name!r} from {dob_column!r}.")
    else:
        result.insert(_position(result, dob_column) + 1, name, age)
        _say(verbose, f"Added {name!r} from {dob_column!r}.")
    return result


# ---------- pipeline ----------

def clean(df: pd.DataFrame, types: StepOptions = None, duplicates: StepOptions = True, missing: StepOptions = True,
          outliers: StepOptions = None, encode: StepOptions = None, scale: StepOptions = None,
          verbose: bool = True) -> pd.DataFrame:
    """Run the cleaning steps in a sensible order and return a new DataFrame.

    Order: types -> duplicates -> missing values -> outliers -> encoding -> normalization.
    The types step (convert_types) is off unless you pass types=True or a dict.
    Each argument is True (use defaults), False/None (skip), or a dict of options
    passed to the matching function, for example:

        clean(df,
              missing={"numeric": "median"},
              outliers={"columns": ["fare"], "action": "cap"},
              encode={"columns": {"sex": "label"}},
              scale={"columns": {"age": "minmax"}})
    """
    def opts(value: StepOptions) -> dict[str, Any]:
        if value is True:
            return {}
        assert isinstance(value, Mapping)
        return dict(value)

    for step, value in (("encode", encode), ("scale", scale)):
        if value is True or (value and "columns" not in value):
            raise ValueError(f"{step} needs the columns to process, e.g. {step}={{'columns': {{...}}}}")

    result = df
    if types:
        result = convert_types(result, verbose=verbose, **opts(types))
    if duplicates:
        result = remove_duplicates(result, verbose=verbose, **opts(duplicates))
    if missing:
        result = fill_missing(result, verbose=verbose, **opts(missing))
    if outliers:
        result = handle_outliers(result, verbose=verbose, **opts(outliers))
    if encode:
        result = encode_categorical(result, verbose=verbose, **opts(encode))
    if scale:
        result = normalize(result, verbose=verbose, **opts(scale))
    return result
