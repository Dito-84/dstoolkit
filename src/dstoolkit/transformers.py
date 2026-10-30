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
    _fill_numeric,
    _position,
)

__all__ = ["MissingFiller", "OutlierCapper", "CategoryEncoder", "Scaler"]

TransformerNumericFill = Literal["mean", "median", "zero"]
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

    Learned: fill_values_ ({column: value}).
    """

    def __init__(self, numeric: TransformerNumericFill | None = "mean",
                 categorical: CategoricalFill | None = "most_frequent",
                 columns: Sequence[str] | None = None) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.columns = columns

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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        result = self._start_transform(X, list(self.fill_values_))
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
      - "label" adds column_label with integer codes in alphabetical order of the
        training categories; missing values stay missing.
    handle_unknown: what to do with a category that wasn't in the training data:
                    "ignore" (default: 0 in every one-hot column, missing label)
                    or "error"
    drop_original: remove the original text columns

    Learned: categories_ ({column: [categories in code order]}).
    """

    def __init__(self, columns: Mapping[str, Encoding] | Sequence[str],
                 handle_unknown: UnknownHandling = "ignore", drop_original: bool = False) -> None:
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.drop_original = drop_original

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
        X = self._start_fit(X)
        _selected(X, list(methods), numeric_only=False)
        self.categories_ = {col: sorted(X[col].dropna().unique(), key=str) for col in methods}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        methods = self._methods()
        result = self._start_transform(X, list(methods))
        for col, method in methods.items():
            known = self.categories_[col]
            unseen = sorted(set(X[col].dropna().unique()) - set(known), key=str)
            if unseen and self.handle_unknown == "error":
                raise ValueError(f"{col!r} has categories that weren't in the training data: {unseen[:5]}")
            at = _position(result, col) + 1
            if method == "onehot":
                for offset, cat in enumerate(known):
                    result.insert(at + offset, f"{col}_{cat}", (X[col] == cat).fillna(False).astype(int).to_numpy())
            else:
                codes = {cat: i for i, cat in enumerate(known)}
                result.insert(at, f"{col}_label", X[col].map(codes).astype("Int64").to_numpy())
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
