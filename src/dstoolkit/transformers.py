"""scikit-learn transformers: learn cleaning values on training data, then apply them anywhere.

The cleaning functions (fill_missing, handle_outliers, encode_categorical,
normalize) learn their numbers from the DataFrame they're given and apply
them in the same call. That's right for exploring a dataset, but before a
model it leaks: means, ranges and outlier bounds computed on all the rows
include the test rows. These transformers split the two steps:

    filler = dst.MissingFiller(numeric="median").fit(train)   # learns from training rows only
    train_clean = filler.transform(train)
    test_clean = filler.transform(test)                       # same values, nothing learned from test

They follow scikit-learn's rules, so they work in a Pipeline and inside
cross-validation, where they are refitted on each training fold:

    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model = make_pipeline(
        dst.MissingFiller(numeric="median", categorical="most_frequent"),
        dst.OutlierCapper(columns=["fare"]),
        dst.CategoryEncoder({"sex": "label", "embarked": "onehot"}, drop_original=True),
        dst.Scaler({"age": "zscore", "fare": "zscore"}),
        LogisticRegression(max_iter=1000),
    )
    cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))

They take and return pandas DataFrames (they need column names), never change
the input, and never add or remove rows: a transformer can't drop rows, so
outliers are capped, not removed. Needs the optional scikit-learn extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
except ImportError as err:
    raise ImportError("dstoolkit's transformers need scikit-learn, an optional extra: pip install scikit-learn, "
                      "or install dstoolkit as dstoolkit[sklearn]") from err

from .cleaning import (
    CATEGORICAL_FILL,
    CategoricalFill,
    Encoding,
    OutlierMethod,
    Scaling,
    _bounds,
    _calendar_days,
    _category_order,
    _fill_numeric,
    _position,
)

__all__ = ["MissingFiller", "OutlierCapper", "CategoryEncoder", "Scaler", "DateFeatures"]

TransformerNumericFill = Literal["mean", "median", "zero"]
IntervalUnit = Literal["days", "calendar_days"]
DatePart = Literal["year", "month", "day", "dayofweek", "dayofyear", "quarter", "hour"]
DATE_PARTS = ("year", "month", "day", "dayofweek", "dayofyear", "quarter", "hour")
_DT_ATTR = {"dayofweek": "day_of_week", "dayofyear": "day_of_year"}   # pandas' current names
UnknownHandling = Literal["ignore", "error"]


class _FrameTransformer(TransformerMixin, BaseEstimator):
    """Shared checks: DataFrame in, DataFrame out, columns checked against what fit() saw."""

    def _start_fit(self, X: Any) -> pd.DataFrame:
        X = self._as_frame(X)
        if X.columns.duplicated().any():
            raise ValueError(f"Column names appear more than once: {sorted(set(X.columns[X.columns.duplicated()]))}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return X

    def _start_transform(self, X: Any, needed: Sequence[Any]) -> pd.DataFrame:
        X = self._as_frame(X)
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise KeyError(f"{type(self).__name__} was fitted with columns that aren't here: {missing}")
        return X.copy()

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} works on pandas DataFrames, because it needs column names; "
                            f"got {type(X).__name__}. Inside a ColumnTransformer, call "
                            f".set_output(transform='pandas') on the ColumnTransformer.")
        return X

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Column names of the transformed DataFrame (scikit-learn's naming hook)."""
        check_is_fitted(self)
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(self._output_names(names), dtype=object)

    def _output_names(self, names: list[Any]) -> list[Any]:
        return names


def _selected(X: pd.DataFrame, columns: Sequence[str] | None, numeric_only: bool) -> list[Any]:
    if columns is None:
        if numeric_only:
            return [c for c in X.columns if _is_number(X[c])]
        return list(X.columns)
    columns = list(columns)
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if numeric_only:
        wrong = [c for c in columns if not _is_number(X[c])]
        if wrong:
            raise TypeError(f"Columns must be numeric: {wrong}")
    return columns


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


class MissingFiller(_FrameTransformer):
    """Fill missing values with values learned from the training data.

    numeric: "mean", "median" or "zero" for number columns, or None to leave them
    categorical: "most_frequent" or "missing" (the word "Missing") for other
                 columns, or None to leave them
    columns: the columns to fill (default: all)

    Like fill_missing(): gaps in date columns are left alone, True/False columns
    keep their gaps with categorical="missing", and an integer column whose fill
    value has decimals is stored as decimals. A column with no values in the
    training data has nothing to learn from and is left as it is. Unlike
    fill_missing(), no rows are dropped. For k-nearest-neighbours filling, use
    scikit-learn's KNNImputer.

    add_indicator: also add a 0/1 column `<column>_missing` that keeps where the
                   gaps were, since filling hides that and a blank can carry
                   signal (a test that wasn't ordered). True flags every selected
                   column that had gaps in the training data; a list flags those
                   columns. The flag goes right after its column. Only flag a
                   column if its blanks were known before the outcome: a blank
                   recorded afterwards leaks the answer.

    Learned: fill_values_ ({column: value}) and indicator_columns_ (the flagged columns).
    """

    def __init__(self, numeric: TransformerNumericFill | None = "mean",
                 categorical: CategoricalFill | None = "most_frequent",
                 columns: Sequence[str] | None = None,
                 add_indicator: bool | Sequence[str] = False) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.columns = columns
        self.add_indicator = add_indicator

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingFiller:
        if self.numeric not in ("mean", "median", "zero", None):
            raise ValueError(f'numeric must be "mean", "median", "zero" or None, got {self.numeric!r}')
        if self.categorical not in CATEGORICAL_FILL:
            raise ValueError(f"categorical must be one of {CATEGORICAL_FILL}, got {self.categorical!r}")
        X = self._start_fit(X)
        values: dict[Any, Any] = {}
        for col in _selected(X, self.columns, numeric_only=False):
            series = X[col].dropna()
            if series.empty or pd.api.types.is_datetime64_any_dtype(X[col]):
                continue
            if _is_number(X[col]):
                if self.numeric is not None:
                    values[col] = {"mean": series.mean(), "median": series.median(), "zero": 0}[self.numeric]
            elif self.categorical == "most_frequent":
                values[col] = series.mode().iloc[0]
            elif self.categorical == "missing" and not isinstance(X[col].dtype, pd.BooleanDtype):
                values[col] = "Missing"
        self.fill_values_ = values
        if isinstance(self.add_indicator, str):
            raise TypeError("add_indicator must be True/False or a list of column names, not one string")
        if self.add_indicator is True:
            flagged = [c for c in _selected(X, self.columns, numeric_only=False) if X[c].isna().any()]
        elif self.add_indicator is False:
            flagged = []
        else:
            flagged = _selected(X, list(self.add_indicator), numeric_only=False)
        clashes = [f"{c}_missing" for c in flagged if f"{c}_missing" in X.columns]
        if clashes:
            raise ValueError(f"Flag column(s) already exist: {clashes}")
        self.indicator_columns_ = flagged
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        result = self._start_transform(X, list(self.fill_values_) + list(self.indicator_columns_))
        for col in self.indicator_columns_:     # before filling, while the gaps are still there
            result.insert(list(result.columns).index(col) + 1, f"{col}_missing", X[col].isna().astype(int))
        for col, value in self.fill_values_.items():
            if not result[col].isna().any():
                continue
            if _is_number(result[col]):
                result[col], _ = _fill_numeric(result[col], value)
            else:
                if isinstance(result[col].dtype, pd.CategoricalDtype) and value not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([value])
                result[col] = result[col].fillna(value)
        return result

    def _output_names(self, names: list[Any]) -> list[Any]:
        flagged = set(getattr(self, "indicator_columns_", []))
        out: list[Any] = []
        for name in names:
            out.append(name)
            if name in flagged:
                out.append(f"{name}_missing")
        return out


class OutlierCapper(_FrameTransformer):
    """Cap outlying values at bounds learned from the training data.

    method: "iqr" (threshold x interquartile range beyond the quartiles, default
            1.5) or "zscore" (threshold standard deviations from the mean, default 3)
    columns: numeric columns to cap (default: every numeric column)

    Values outside the bounds are pulled back to them; rows are kept, since a
    transformer can't remove rows. Learned: bounds_ ({column: (low, high)}).
    """

    def __init__(self, columns: Sequence[str] | None = None, method: OutlierMethod = "iqr",
                 threshold: float | None = None) -> None:
        self.columns = columns
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> OutlierCapper:
        X = self._start_fit(X)
        self.bounds_ = {col: tuple(float(b) for b in _bounds(X[col], self.method, self.threshold))
                        for col in _selected(X, self.columns, numeric_only=True) if X[col].notna().any()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        result = self._start_transform(X, list(self.bounds_))
        for col, (low, high) in self.bounds_.items():
            result[col] = result[col].clip(lower=low, upper=high)
        return result


class CategoryEncoder(_FrameTransformer):
    """Turn text categories into numbers, using the categories seen in the training data.

    columns: {column: "onehot" or "label"}, or a list of columns to one-hot encode
      - "onehot" adds a 0/1 column per training category (sex_female, sex_male).
        Missing values are 0 in every column.
      - "label" adds column_label with integer codes in order of the training
        categories: numeric order when they're all numbers (1, 2, 10), otherwise
        alphabetical; missing values stay missing.
    handle_unknown: what to do with a category that wasn't in the training data:
                    "ignore" (default: 0 in every one-hot column, missing label)
                    or "error"
    drop_original: remove the original text columns
    min_frequency: group rare categories into one, `other_label` ("Other"):
                   a whole number is the fewest training rows a category needs,
                   a fraction is the smallest share of the column's values (0.01
                   = 1%). Codes, providers and ZIP codes often have hundreds of
                   values, mostly rare; one column per rare value adds little but
                   noise. When there is an "Other" group, categories that weren't
                   in the training data join it instead of being ignored.
    other_label: the name of the group (a real category with this name joins it)

    Learned: categories_ ({column: [categories in code order]}) and rare_
    ({column: [categories grouped into other_label]}).
    """

    def __init__(self, columns: Mapping[str, Encoding] | Sequence[str],
                 handle_unknown: UnknownHandling = "ignore", drop_original: bool = False,
                 min_frequency: float | None = None, other_label: str = "Other") -> None:
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.drop_original = drop_original
        self.min_frequency = min_frequency
        self.other_label = other_label

    def _methods(self) -> dict[Any, str]:
        if isinstance(self.columns, Mapping):
            return dict(self.columns)
        if isinstance(self.columns, str):
            raise TypeError("columns must be a list of column names or a {column: method} dict, not one string")
        return {c: "onehot" for c in self.columns}

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoryEncoder:
        methods = self._methods()
        bad = {c: m for c, m in methods.items() if m not in ("onehot", "label")}
        if bad:
            raise ValueError(f'Encoding must be "onehot" or "label": {bad}')
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError(f'handle_unknown must be "ignore" or "error", got {self.handle_unknown!r}')
        m = self.min_frequency
        if m is not None and (isinstance(m, bool) or not isinstance(m, (int, float)) or m <= 0
                              or (isinstance(m, float) and m >= 1)):
            raise ValueError("min_frequency must be a whole number of rows (1 or more) or a share between 0 and 1")
        X = self._start_fit(X)
        _selected(X, list(methods), numeric_only=False)
        self.categories_, self.rare_ = {}, {}
        for col in methods:
            counts = X[col].dropna().value_counts()
            needed = 0 if m is None else (m * counts.sum() if isinstance(m, float) else m)
            rare = _category_order(counts[counts < needed].index)
            frequent = _category_order([c for c in counts.index if c not in rare and c != self.other_label])
            self.rare_[col] = rare
            self.categories_[col] = frequent + ([self.other_label] if rare or self.other_label in counts.index else [])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        methods = self._methods()
        result = self._start_transform(X, list(methods))
        for col, method in methods.items():
            known = self.categories_[col]
            values = X[col]
            if self.rare_.get(col):    # rare and unseen categories join the "Other" group
                values = values.where(values.isna() | values.isin(known), self.other_label)
                if isinstance(values.dtype, pd.CategoricalDtype):
                    values = values.astype(object)
            unseen = sorted(set(X[col].dropna().unique()) - set(known) - set(self.rare_.get(col, [])), key=str)
            if self.rare_.get(col):
                unseen = []            # they're in the "Other" group
            if unseen and self.handle_unknown == "error":
                raise ValueError(f"{col!r} has categories that weren't in the training data: {unseen[:5]}")
            at = _position(result, col) + 1
            if method == "onehot":
                for offset, cat in enumerate(known):
                    result.insert(at + offset, f"{col}_{cat}", (values == cat).fillna(False).astype(int).to_numpy())
            else:
                codes = {cat: i for i, cat in enumerate(known)}
                result.insert(at, f"{col}_label", values.map(codes).astype("Int64").to_numpy())
        if self.drop_original:
            result = result.drop(columns=list(methods))
        return result

    def _output_names(self, names: list[Any]) -> list[Any]:
        methods = self._methods()
        out: list[Any] = []
        for name in names:
            if name not in methods or not self.drop_original:
                out.append(name)
            if name in methods:
                out += ([f"{name}_{cat}" for cat in self.categories_[name]] if methods[name] == "onehot"
                        else [f"{name}_label"])
        return out


class Scaler(_FrameTransformer):
    """Rescale numeric columns with the ranges or averages of the training data.

    columns: {column: "minmax" or "zscore"}, or a list of columns to min-max scale
      - "minmax": (x - training min) / (training max - training min); new data
        outside the training range goes below 0 or above 1
      - "zscore": (x - training mean) / training standard deviation

    Missing values stay missing. A column with one distinct value in training
    becomes 0, as in normalize(). Learned: stats_ ({column: (centre, spread)}).
    """

    def __init__(self, columns: Mapping[str, Scaling] | Sequence[str]) -> None:
        self.columns = columns

    def _methods(self) -> dict[Any, str]:
        if isinstance(self.columns, Mapping):
            return dict(self.columns)
        if isinstance(self.columns, str):
            raise TypeError("columns must be a list of column names or a {column: method} dict, not one string")
        return {c: "minmax" for c in self.columns}

    def fit(self, X: pd.DataFrame, y: Any = None) -> Scaler:
        methods = self._methods()
        bad = {c: m for c, m in methods.items() if m not in ("minmax", "zscore")}
        if bad:
            raise ValueError(f'Scaling must be "minmax" or "zscore": {bad}')
        X = self._start_fit(X)
        _selected(X, list(methods), numeric_only=True)
        stats: dict[Any, tuple[float, float]] = {}
        for col, method in methods.items():
            values = X[col].astype(float)
            if method == "minmax":
                stats[col] = (float(values.min()), float(values.max() - values.min()))
            else:
                stats[col] = (float(values.mean()), float(values.std(ddof=0)))
        self.stats_ = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        result = self._start_transform(X, list(self.stats_))
        for col, (centre, spread) in self.stats_.items():
            values = result[col].astype(float)
            result[col] = (values - centre) / spread if spread else values * 0
        return result


def _completed_years(born: pd.Series, on: pd.Series) -> pd.Series:
    """Whole years from born to on; missing when either is missing or born is later."""
    years = on.dt.year - born.dt.year
    before_birthday = (on.dt.month < born.dt.month) | ((on.dt.month == born.dt.month) & (on.dt.day < born.dt.day))
    age = (years - before_birthday.astype(int)).astype("float")
    return age.where(born.notna() & on.notna() & (born <= on))


class DateFeatures(_FrameTransformer):
    """Turn date columns into numbers a model can use.

    columns: date columns to split into parts (default: every date column)
    parts: which parts become columns, named <column>_<part>: "year", "month",
           "day", "dayofweek" (Monday = 0), "dayofyear", "quarter", "hour"
    intervals: {new column: (start column, end column)}: the days from start to
               end, e.g. {"stay_days": ("admitted", "discharged")}. Durations
               often matter more than the dates themselves.
    interval_unit: how intervals are counted. "days" (default): elapsed time in
                   days, with fractions (23:00 to 01:00 the next day is 0.083).
                   "calendar_days": whole calendar days, the same as
                   add_days_between() (that example is 1, and a same-day
                   discharge 0); dates with a time zone are counted on their
                   local calendar.
    ages: {new column: (birth date column, date column)}: age in completed
          years on the second date, e.g. {"age_at_admission": ("date_of_birth",
          "admitted")}. A birth date after the second date gives a missing age,
          not a negative one.
    drop_original: remove the date columns afterwards (default True: most models
                   can't take dates)

    Columns must already be dates (dst.to_datetime_columns() converts text).
    Nothing is learned from the data, so new data is treated exactly like the
    training data; missing dates give missing features.

    Learned: date_columns_ (the columns split into parts).
    """

    def __init__(self, columns: Sequence[str] | None = None,
                 parts: Sequence[DatePart] = ("year", "month", "dayofweek"),
                 intervals: Mapping[str, tuple[str, str]] | None = None, drop_original: bool = True,
                 ages: Mapping[str, tuple[str, str]] | None = None,
                 interval_unit: IntervalUnit = "days") -> None:
        self.columns = columns
        self.parts = parts
        self.intervals = intervals
        self.drop_original = drop_original
        self.ages = ages
        self.interval_unit = interval_unit

    def _pairs(self) -> dict[str, tuple[str, str]]:
        return {**dict(self.intervals or {}), **dict(self.ages or {})}

    def fit(self, X: pd.DataFrame, y: Any = None) -> DateFeatures:
        bad = [p for p in self.parts if p not in DATE_PARTS]
        if isinstance(self.parts, str) or bad:
            raise ValueError(f"parts must be a list drawn from {DATE_PARTS}")
        if self.interval_unit not in ("days", "calendar_days"):
            raise ValueError(f'interval_unit must be "days" or "calendar_days", got {self.interval_unit!r}')
        X = self._start_fit(X)
        if self.columns is None:
            dates = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        else:
            dates = _selected(X, self.columns, numeric_only=False)
        same = set(self.intervals or {}) & set(self.ages or {})
        if same:
            raise ValueError(f"Names used for both an interval and an age: {sorted(same)}")
        used = dates + [c for pair in self._pairs().values() for c in pair]
        missing = [c for c in used if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        not_dates = sorted({str(c) for c in used if not pd.api.types.is_datetime64_any_dtype(X[c])})
        if not_dates:
            raise TypeError(f"Not date columns: {not_dates}. Convert them first with dst.to_datetime_columns()")
        clashes = [n for n in self._new_names(dates) if n in X.columns]
        if clashes:
            raise ValueError(f"Columns already exist: {clashes}")
        self.date_columns_ = dates
        return self

    def _new_names(self, dates: Sequence[Any]) -> list[str]:
        return [f"{c}_{p}" for c in dates for p in self.parts] + list(self._pairs())

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        pairs = self._pairs()
        result = self._start_transform(X, self.date_columns_ + [c for pair in pairs.values() for c in pair])
        for col in self.date_columns_:
            values = pd.to_datetime(X[col])
            at = list(result.columns).index(col) + 1
            for offset, part in enumerate(self.parts):
                piece = getattr(values.dt, _DT_ATTR.get(part, part)).astype("float")
                result.insert(at + offset, f"{col}_{part}", piece.to_numpy())
        for name, (start, end) in (self.intervals or {}).items():
            first, last = pd.to_datetime(X[start]), pd.to_datetime(X[end])
            if self.interval_unit == "calendar_days":
                result[name] = _calendar_days(first, last).astype("float").to_numpy()
            else:
                result[name] = ((last - first).dt.total_seconds() / 86400).to_numpy()
        for name, (born, on) in (self.ages or {}).items():
            result[name] = _completed_years(pd.to_datetime(X[born]), pd.to_datetime(X[on])).to_numpy()
        if self.drop_original:
            used = set(self.date_columns_) | {c for p in pairs.values() for c in p}
            result = result.drop(columns=[c for c in result.columns if c in used])
        return result

    def _output_names(self, names: list[Any]) -> list[Any]:
        pairs = self._pairs()
        used = set(self.date_columns_) | {c for p in pairs.values() for c in p}
        out: list[Any] = []
        for name in names:
            if not (self.drop_original and name in used):
                out.append(name)
            if name in self.date_columns_:
                out += [f"{name}_{p}" for p in self.parts]
        return out + list(pairs)
