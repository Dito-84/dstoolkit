"""Data cleaning and preprocessing helpers.

Each function takes a DataFrame and returns a new one; the input is never
modified. Steps can be used on their own or chained with `clean()`:

    import preprocessing as pp

    df = pp.remove_duplicates(df)
    df = pp.fill_missing(df, numeric="median")
    df = pp.handle_outliers(df, columns=["fare"], method="iqr", action="cap")
    df = pp.encode_categorical(df, {"sex": "label", "embarked": "onehot"})
    df = pp.normalize(df, {"age": "minmax", "fare": "zscore"})

    # or all at once
    df = pp.clean(df, missing={"numeric": "median"}, outliers={"columns": ["fare"], "action": "cap"})

Pass `verbose=False` to any function to silence its summary message.
"""

import warnings

import numpy as np
import pandas as pd

__all__ = [
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

DOB_NAME_HINTS = ("dob", "date_of_birth", "dateofbirth", "birth_date", "birthdate", "birthday")


def _say(verbose, message):
    if verbose:
        print(message)


def _check_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")


def _numeric_columns(df, columns):
    """Requested columns (default: every numeric column), checked to be numeric."""
    if columns is None:
        return list(df.select_dtypes(include="number").columns)
    columns = list(columns)
    _check_columns(df, columns)
    not_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        raise TypeError(f"Columns must be numeric: {not_numeric}")
    return columns


# ---------- duplicates ----------

def remove_duplicates(df, subset=None, keep="first", verbose=True):
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


def fill_missing(df, numeric="mean", categorical="most_frequent", columns=None, n_neighbors=5,
                 drop_empty_rows=True, verbose=True):
    """Fill missing values, with separate methods for numeric and text columns.

    columns: only fill these columns (default: all). Useful for leaving mostly
             empty columns alone rather than inventing values for them.

    numeric: 'mean', 'median', 'knn' (estimate from the n_neighbors most similar
             rows), 'zero', or None to leave numeric gaps alone
    categorical: 'most_frequent', 'missing' (fill with the word "Missing"),
                 or None to leave them alone
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

    if cat_cols and categorical is not None:
        for col in cat_cols:
            if categorical == "most_frequent":
                mode = result[col].mode(dropna=True)
                if mode.empty:
                    continue  # column is entirely empty; nothing to copy from
                result[col] = result[col].fillna(mode.iloc[0])
            else:
                result[col] = result[col].fillna("Missing")

    done = {c: int(filled[c] - result[c].isna().sum()) for c in num_cols + cat_cols}
    done = {c: n for c, n in done.items() if n}
    _say(verbose, "Filled missing values: " + ", ".join(f"{c} ({n})" for c, n in done.items())
         if done else "No missing values filled.")
    return result


# ---------- outliers ----------

def _bounds(series, method, threshold):
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


def find_outliers(df, columns=None, method="iqr", threshold=None):
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


def handle_outliers(df, columns=None, method="iqr", threshold=None, action="remove", verbose=True):
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


def winsorize(df, columns=None, lower=0.05, upper=0.95, verbose=True):
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


def log_transform(df, columns, suffix="", verbose=True):
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

def encode_categorical(df, columns, onehot_limit=10, drop_original=False, verbose=True):
    """Turn text categories into numbers.

    columns: {column: 'onehot' or 'label'}, or a list of columns to one-hot encode
      - 'onehot' adds one 0/1 column per category (e.g. sex_female, sex_male)
      - 'label' adds a column of integer codes (column_label), in alphabetical
        order of the categories; missing values stay missing
    onehot_limit: warn when one-hot encoding would add more columns than this
    drop_original: remove the original text columns after encoding
    """
    if isinstance(columns, (list, tuple)):
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
            at = result.columns.get_loc(col) + 1
            for offset, name in enumerate(dummies.columns):
                result.insert(at + offset, name, dummies[name])
        else:
            categories = sorted(df[col].dropna().unique(), key=str)
            codes = {cat: i for i, cat in enumerate(categories)}
            result.insert(result.columns.get_loc(col) + 1, f"{col}_label", df[col].map(codes).astype("Int64"))
            _say(verbose, f"{col} labels: " + ", ".join(f"{k}={v}" for k, v in codes.items()))
    if drop_original:
        result = result.drop(columns=list(columns))
    _say(verbose, "Encoded " + ", ".join(f"{c} ({m})" for c, m in columns.items()) + ".")
    return result


# ---------- normalization ----------

def normalize(df, columns, verbose=True):
    """Rescale numeric columns.

    columns: {column: 'minmax' or 'zscore'}, or a list of columns to min-max scale
      - 'minmax' rescales to the 0–1 range
      - 'zscore' centres on 0 with a standard deviation of 1
    Missing values stay missing. A column with only one distinct value becomes 0.
    """
    if isinstance(columns, (list, tuple)):
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

def add_age_column(df, dob_column=None, as_of=None, name="age", verbose=True):
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
        result.insert(result.columns.get_loc(dob_column) + 1, name, age)
        _say(verbose, f"Added {name!r} from {dob_column!r}.")
    return result


# ---------- pipeline ----------

def clean(df, duplicates=True, missing=True, outliers=None, encode=None, scale=None, verbose=True):
    """Run the cleaning steps in a sensible order and return a new DataFrame.

    Order: duplicates -> missing values -> outliers -> encoding -> normalization.
    Each argument is True (use defaults), False/None (skip), or a dict of options
    passed to the matching function, for example:

        clean(df,
              missing={"numeric": "median"},
              outliers={"columns": ["fare"], "action": "cap"},
              encode={"columns": {"sex": "label"}},
              scale={"columns": {"age": "minmax"}})
    """
    def opts(value):
        return {} if value is True else dict(value)

    for step, value in (("encode", encode), ("scale", scale)):
        if value is True or (value and "columns" not in value):
            raise ValueError(f"{step} needs the columns to process, e.g. {step}={{'columns': {{...}}}}")

    result = df
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
