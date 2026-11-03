"""Monitoring: has new data drifted from the data a model was built on?

    import dstoolkit as dst

    changes = dst.compare(train, new)       # one row per column, biggest changes first
    changes[changes["status"] != "stable"]

For every column in either DataFrame, compare() checks:

    type        did the kind of data change (numbers arriving as text, say)?
    shift       how far the distribution moved: the population stability index
                (PSI), over the reference data's deciles for numbers and over
                the categories for everything else
    blanks      did the share of missing values change?
    categories  are there values the reference data never had (a new code, a
                new spelling)?
    columns     was a column added or removed?

Each row gets a status: "major", "moderate" or "stable" (or "added" /
"removed"). The PSI thresholds are the usual rule of thumb: below 0.1 stable,
0.1 to 0.25 moderate, 0.25 or more major. A p-value (Kolmogorov-Smirnov for
numbers, chi-square for categories, adjusted for testing many columns) is
reported too, but doesn't set the status: with thousands of rows, tiny,
harmless shifts are "significant".

Dates are expected to move forward, so their distribution isn't compared;
identifiers and free text (many values, mostly distinct) only get the type
and blank checks. Nothing is changed in either DataFrame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._messages import say
from .cleaning import DATE_PATTERN

__all__ = ["compare"]

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
BLANKS_MODERATE = 0.02      # change in the share of blanks (2 percentage points)
BLANKS_MAJOR = 0.10
NEW_CATEGORIES_MODERATE = 0.02  # share of new rows in categories the reference never had
NEW_CATEGORIES_MAJOR = 0.05
RARE_SHARE = 0.01           # categories rarer than this in the reference share one bin, so the PSI isn't noise
ID_SHARE = 0.2              # text with more distinct values than max_categories AND this share of rows: an identifier
FEW_VALUES = 10             # numbers with this many distinct values or fewer are compared as categories
_EPS = 1e-4                 # stands in for an empty bin, so the PSI stays finite

COMPARE_COLUMNS = ["column", "status", "kind", "shift_psi", "p_adjusted", "blanks_before", "blanks_after",
                   "new_categories", "new_category_share", "issues", "rows_before", "rows_after"]
_RANK = {"major": 0, "added": 0, "removed": 0, "moderate": 1, "stable": 2}


def compare(reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str] | None = None,
            max_categories: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Compare new data with the data a model was built on, column by column.

    reference: the data the model learned from (e.g. the training data)
    current: the new data (e.g. this month's records)
    columns: only compare these (default: every column in either DataFrame)
    max_categories: text columns with more distinct values than this, that are
                    also distinct in at least 20% of rows, are treated as
                    identifiers or free text (patient IDs, names, notes): only
                    their type and blanks are compared. Codes with many values
                    that repeat (diagnosis codes) are still compared.

    Returns one row per column, the biggest changes first: status ("major",
    "moderate", "stable", "added" or "removed"), the kind of data, the shift
    (PSI), an adjusted p-value, the share of blanks before and after, categories
    the reference never had and the share of new rows in them, and the issues in
    words. See the module docstring for the thresholds.
    """
    for name, frame in (("reference", reference), ("current", current)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame, got {type(frame).__name__}")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name} has repeated column names; rename them first")
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    names = list(dict.fromkeys([*reference.columns, *current.columns])) if columns is None else list(columns)
    unknown = [c for c in names if c not in reference.columns and c not in current.columns]
    if unknown:
        raise KeyError(f"Columns not found in either DataFrame: {unknown}")

    rows: list[dict[str, Any]] = []
    for col in names:
        if col not in current.columns:
            rows.append(_row(col, "removed", _kind(reference[col], max_categories), len(reference), len(current),
                             issues=["missing from the new data"]))
        elif col not in reference.columns:
            rows.append(_row(col, "added", _kind(current[col], max_categories), len(reference), len(current),
                             issues=["not in the reference data"]))
        else:
            rows.append(_compare_column(col, reference[col], current[col], max_categories))

    result = pd.DataFrame(rows, columns=[*COMPARE_COLUMNS, "p_value"])
    tested = result["p_value"].notna()
    result["p_adjusted"] = np.nan
    if tested.any():
        result.loc[tested, "p_adjusted"] = _benjamini_hochberg(result.loc[tested, "p_value"].to_numpy(dtype=float))
    order = result["status"].map(_RANK).to_numpy() * 10 - result["shift_psi"].fillna(0).to_numpy()
    result = result.iloc[np.argsort(order, kind="stable")].drop(columns="p_value").reset_index(drop=True)

    counts = result["status"].value_counts()
    parts = [f"{counts.get(s, 0)} {s}" for s in ("major", "moderate", "stable") if counts.get(s, 0)]
    for s in ("added", "removed"):
        if counts.get(s, 0):
            parts.append(f"{s}: {', '.join(str(c) for c in result.loc[result['status'] == s, 'column'])}")
    say(verbose, f"Compared {len(result)} column(s): " + "; ".join(parts) + ".")
    return result


def _row(col: Any, status: str, kind: str, n_before: int, n_after: int, **values: Any) -> dict[str, Any]:
    issues = values.pop("issues", [])
    return {"column": col, "status": status, "kind": kind, "shift_psi": np.nan, "p_adjusted": np.nan,
            "blanks_before": np.nan, "blanks_after": np.nan, "new_categories": [], "new_category_share": 0.0,
            "issues": "; ".join(issues), "rows_before": n_before, "rows_after": n_after, "p_value": np.nan,
            **values}


def _kind(series: pd.Series, max_categories: int) -> str:
    present = series.dropna()
    if present.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "category"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    text = present.astype(str)
    sample = text.sample(min(len(text), 500), random_state=0)
    if sample.str.match(DATE_PATTERN).mean() >= 0.9:
        return "date"
    distinct = text.nunique()
    if distinct > max_categories and distinct >= ID_SHARE * len(text):
        return "identifier or free text"
    return "category"


def _compare_column(col: Any, before: pd.Series, after: pd.Series, max_categories: int) -> dict[str, Any]:
    kind_before, kind_after = _kind(before, max_categories), _kind(after, max_categories)
    blanks_before = float(before.isna().mean()) if len(before) else 0.0
    blanks_after = float(after.isna().mean()) if len(after) else 0.0
    issues: list[str] = []
    severity = "stable"

    def worse(level: str) -> None:
        nonlocal severity
        if _RANK[level] < _RANK[severity]:
            severity = level

    if "empty" not in (kind_before, kind_after) and kind_before != kind_after:
        issues.append(f"type changed: {kind_before} → {kind_after}")
        worse("major")

    change = blanks_after - blanks_before
    if abs(change) >= BLANKS_MODERATE:
        issues.append(f"blanks {blanks_before:.1%} → {blanks_after:.1%}")
        worse("major" if abs(change) >= BLANKS_MAJOR else "moderate")

    psi, p_value, new_share = np.nan, np.nan, 0.0
    new: list[Any] = []
    kind = kind_after if kind_before == "empty" else kind_before
    if kind_before == kind_after and kind in ("number", "category"):
        x, y = before.dropna(), after.dropna()
        if len(x) and len(y):
            if kind == "number":
                x, y = x.astype(float), y.astype(float)    # 1 and 1.0 are the same value
            few = kind == "number" and x.nunique() <= FEW_VALUES
            if kind == "number" and not few:
                psi, p_value = _numeric_shift(x.astype(float), y.astype(float))
            else:
                psi, p_value, new, new_share = _category_shift(x, y)
                if kind == "number":
                    new, new_share = [], 0.0          # a new count or amount is a shift, measured by the PSI
            if psi >= PSI_MAJOR:
                issues.append(f"distribution shifted (PSI {psi:.2f})")
                worse("major")
            elif psi >= PSI_MODERATE:
                issues.append(f"distribution shifted (PSI {psi:.2f})")
                worse("moderate")
            if new:
                shown = ", ".join(str(v) for v in new[:5]) + (", ..." if len(new) > 5 else "")
                issues.append(f"{len(new)} new value(s), {new_share:.1%} of new rows: {shown}")
                if new_share >= NEW_CATEGORIES_MAJOR:
                    worse("major")
                elif new_share >= NEW_CATEGORIES_MODERATE:
                    worse("moderate")
    elif kind == "date":
        issues.append("dates: expected to move, distribution not compared")
    elif kind == "identifier or free text":
        issues.append("identifier or free text: distribution not compared")

    return _row(col, severity, kind if kind_before == kind_after else f"{kind_before} → {kind_after}",
                len(before), len(after), shift_psi=psi, p_value=p_value, blanks_before=blanks_before,
                blanks_after=blanks_after, new_categories=new, new_category_share=new_share, issues=issues)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _EPS, None)
    a = np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_shift(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    from scipy import stats
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    before = np.histogram(x, edges)[0] / len(x)
    after = np.histogram(y, edges)[0] / len(y)
    return _psi(before, after), float(stats.ks_2samp(x, y).pvalue)


def _category_shift(x: pd.Series, y: pd.Series) -> tuple[float, float, list[Any], float]:
    from scipy import stats
    before, after = x.astype(str).value_counts(), y.astype(str).value_counts()
    new = [c for c in after.index if c not in before.index]           # most frequent first
    new_share = float(after[new].sum() / len(y))
    # Rare reference categories and new ones share one bin: a long tail of rare codes would
    # otherwise make the PSI move by chance alone.
    common = [c for c in before.index if before[c] >= RARE_SHARE * len(x)]
    other = "\0rare or new"
    before = before.groupby(lambda c: c if c in common else other).sum()
    after = after.groupby(lambda c: c if c in common else other).sum()
    cats = sorted(set(before.index) | set(after.index))
    b, a = before.reindex(cats, fill_value=0), after.reindex(cats, fill_value=0)
    p_value = np.nan
    if len(cats) > 1:
        with np.errstate(all="ignore"):
            p_value = float(stats.chi2_contingency(np.vstack([b.to_numpy(), a.to_numpy()]))[1])
    return _psi(b.to_numpy() / b.sum(), a.to_numpy() / a.sum()), p_value, new, new_share


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty(n)
    out[order] = adjusted
    return out
