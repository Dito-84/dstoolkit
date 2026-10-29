"""Validation: check a DataFrame against rules and report every problem.

Nothing is changed or converted: you get a report of which rows break which
rule, with examples, and decide what to do about it.

    import dstoolkit as dst

    report = dst.validate(df, {
        "age": {"type": "number", "min": 0, "max": 120, "required": True},
        "sex": {"allowed": ["male", "female"]},
        "ticket_id": {"unique": True, "pattern": r"T\\d{4}"},
    })
    report.ok                 # True when every rule passed
    report.summary()          # one row per rule: checked, failed, examples
    report.failures()         # one row per failing value, with the reason
    report.raise_if_failed()  # stop here if anything failed

    # validate what clean() produced
    dst.validate(dst.clean(df), rules).raise_if_failed()

Rules per column (use any combination):

    type      "number", "integer", "text", "boolean", "datetime" or "category"
    required  True: no blank values (missing, or empty text)
    min, max  lowest / highest allowed value: numbers, or dates for date columns
    allowed   the only values permitted, e.g. ["male", "female"]
    unique    True: no value appears more than once
    pattern   regular expression the whole value must match

Missing values are skipped by every rule except `required`. Values are never
converted before checking: the text "12" is not a number, and 1 is not True.

The rule format is experimental until dstoolkit 1.0 and may change.
"""

from __future__ import annotations

import datetime as dt
import difflib
import html
import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

__all__ = ["validate", "ValidationReport", "RuleResult", "ColumnRule", "ValidationError"]

TypeName = Literal["number", "integer", "text", "boolean", "datetime", "category"]
TYPE_NAMES: tuple[str, ...] = ("number", "integer", "text", "boolean", "datetime", "category")


class ColumnRule(TypedDict, total=False):
    """The checks for one column. Type checkers flag misspelled keys."""
    type: TypeName
    required: bool
    min: Any
    max: Any
    allowed: Sequence[Any]
    unique: bool
    pattern: str


RULE_KEYS: tuple[str, ...] = ("type", "required", "min", "max", "allowed", "unique", "pattern")
PRINTED_EXAMPLES = 3   # examples per failed rule in the printed summary


class ValidationError(ValueError):
    """Raised by `ValidationReport.raise_if_failed()`. The report is in `.report`."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.message())
        self.report = report


@dataclass(frozen=True)
class RuleResult:
    """The outcome of one rule on one column."""
    column: str
    rule: str          # "type", "required", "min", "max", "allowed", "unique", "pattern" or "column"
    expected: str      # what the rule asks for, in words
    checked: int       # how many values were checked
    failed: int        # how many rows failed
    note: str = ""     # a problem with the whole column (not found, wrong storage type)
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp), repr=False)
    values: list[Any] = field(default_factory=list, repr=False)
    reasons: list[str] = field(default_factory=list, repr=False)

    @property
    def passed(self) -> bool:
        return self.failed == 0 and not self.note


class ValidationReport:
    """The result of `validate()`. Nothing in the data was changed."""

    def __init__(self, n_rows: int, index: pd.Index, results: list[RuleResult], max_examples: int) -> None:
        self.n_rows = n_rows
        self.results = results
        self.max_examples = max_examples
        self._index = index

    @property
    def ok(self) -> bool:
        """True when every rule passed."""
        return all(r.passed for r in self.results)

    @property
    def failed_rules(self) -> list[RuleResult]:
        return [r for r in self.results if not r.passed]

    @property
    def failing_index(self) -> pd.Index:
        """Labels of rows that failed any rule, in row order: `df.loc[report.failing_index]`."""
        positions = [r.rows for r in self.results if r.rows.size]
        if not positions:
            return self._index[:0]
        return self._index[np.unique(np.concatenate(positions))]

    def summary(self) -> pd.DataFrame:
        """One row per rule: what was checked, how many failed, and a few examples."""
        return pd.DataFrame([{
            "column": r.column, "rule": r.rule, "expected": r.expected,
            "checked": r.checked, "failed": r.failed,
            "failed_%": round(100 * r.failed / r.checked, 1) if r.checked else 0.0,
            "passed": r.passed,
            "examples": r.note or self._examples(r, self.max_examples),
        } for r in self.results], columns=["column", "rule", "expected", "checked", "failed",
                                            "failed_%", "passed", "examples"])

    def failures(self, column: str | None = None, rule: str | None = None) -> pd.DataFrame:
        """Every failing value: row label, column, rule, value and reason.

        Filter with `column=` and/or `rule=`. Problems with a whole column (not
        found, wrong storage type) are in `summary()`, not here.
        """
        frames = []
        for r in self.results:
            if not r.rows.size or (column is not None and r.column != column) or (rule is not None and r.rule != rule):
                continue
            frames.append(pd.DataFrame({
                "row": self._index[r.rows], "column": r.column, "rule": r.rule,
                "value": pd.Series(r.values, dtype=object), "reason": r.reasons,
            }))
        if not frames:
            return pd.DataFrame({"row": pd.Series(dtype=object), "column": pd.Series(dtype=object),
                                 "rule": pd.Series(dtype=object), "value": pd.Series(dtype=object),
                                 "reason": pd.Series(dtype=object)})
        return pd.concat(frames, ignore_index=True)

    def raise_if_failed(self) -> ValidationReport:
        """Raise `ValidationError` if any rule failed; otherwise return the report."""
        if not self.ok:
            raise ValidationError(self)
        return self

    def message(self) -> str:
        """The printed summary: one line overall, plus one per failed rule."""
        failed = self.failed_rules
        head = f"Validated {self.n_rows:,} rows against {len(self.results)} rule(s): "
        if not failed:
            return head + "all passed."
        lines = [head + f"{len(failed)} failed ({len(self.failing_index):,} rows)."]
        for r in failed:
            if r.note:
                lines.append(f"  {r.column}: {r.note}")
            else:
                lines.append(f"  {r.column}: {r.rule} ({r.expected}) failed for {r.failed:,} row(s), "
                             f"e.g. {self._examples(r, PRINTED_EXAMPLES)}")
        return "\n".join(lines)

    def _examples(self, r: RuleResult, limit: int) -> str:
        return "; ".join(f"row {self._index[pos]!r} = {_show(value)}"
                         for pos, value in zip(r.rows[:limit], r.values[:limit]))

    def _repr_html_(self) -> str:
        return f"<p>{_escape(self.message().splitlines()[0])}</p>" + self.summary().to_html(index=False)

    def __repr__(self) -> str:
        return (f"<ValidationReport: {self.n_rows:,} rows, {len(self.results)} rule(s), "
                f"{len(self.failed_rules)} failed>")


def validate(df: pd.DataFrame, rules: Mapping[Hashable, ColumnRule], max_examples: int = 5,
             verbose: bool = True) -> ValidationReport:
    """Check `df` against `rules` and report every problem. The data is not changed.

    rules: {column: {rule: value}}; see the module docstring for the rule names
    max_examples: example rows shown per rule in `summary()` (counts are always exact)

    A column that doesn't exist is reported as a failure (with the closest
    matching name). A mistake in the rules themselves, such as a misspelled
    rule name or an invalid regular expression, raises an error before
    anything is checked.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate() needs a DataFrame, got {type(df).__name__}")
    if max_examples < 0:
        raise ValueError("max_examples can't be negative")
    plans = _parse_rules(rules)

    names = [str(c) for c in df.columns]
    results: list[RuleResult] = []
    for column, checks in plans:
        label = str(column)
        if column not in df.columns:
            close = difflib.get_close_matches(label, names, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            results.append(RuleResult(label, "column", "column exists", 0, 0, note=f"column not found{hint}"))
            continue
        if (df.columns == column).sum() > 1:
            raise ValueError(f"Column name {label!r} appears more than once; rename duplicates first")
        series = df[column]
        for key, value in checks:
            results.append(_run(series, label, key, value))

    report = ValidationReport(len(df), df.index, results, max_examples)
    if verbose:
        print(report.message())
    return report


# ---------- reading the rules ----------

def _parse_rules(rules: Mapping[Hashable, ColumnRule]) -> list[tuple[Hashable, list[tuple[str, Any]]]]:
    """Check the rules are well formed, so mistakes raise before any data is read."""
    if not isinstance(rules, Mapping):
        raise TypeError("rules must be a dict of {column: {rule: value}}")
    plans = []
    for column, rule in rules.items():
        if not isinstance(rule, Mapping):
            raise TypeError(f"Rules for {column!r} must be a dict, e.g. {{'required': True}}")
        if not rule:
            raise ValueError(f"No rules given for {column!r}")
        unknown = [k for k in rule if k not in RULE_KEYS]
        if unknown:
            hints = [f"{k!r} (did you mean {m[0]!r}?)" if (m := difflib.get_close_matches(str(k), RULE_KEYS, n=1))
                     else repr(k) for k in unknown]
            raise ValueError(f"Unknown rule(s) for {column!r}: {', '.join(hints)}. "
                             f"Valid rules: {', '.join(RULE_KEYS)}")
        checks: list[tuple[str, Any]] = []
        for key in RULE_KEYS:
            if key not in rule:
                continue
            value: Any = rule[key]  # type: ignore[literal-required]
            if key == "type" and value not in TYPE_NAMES:
                raise ValueError(f"{column!r}: type must be one of {', '.join(TYPE_NAMES)}, got {value!r}")
            if key in ("required", "unique"):
                if not isinstance(value, bool):
                    raise TypeError(f"{column!r}: {key} must be True or False, got {value!r}")
                if not value:
                    continue
            if key in ("min", "max"):
                value = _bound(column, key, value)
            if key == "allowed":
                if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (Sequence, set, frozenset)):
                    raise TypeError(f"{column!r}: allowed must be a list of values, e.g. ['male', 'female']")
                value = list(value)
                if not value:
                    raise ValueError(f"{column!r}: allowed is empty, so every value would fail")
            if key == "pattern":
                if not isinstance(value, str):
                    raise TypeError(f"{column!r}: pattern must be a regular expression string")
                try:
                    value = re.compile(value)
                except re.error as err:
                    raise ValueError(f"{column!r}: invalid pattern {rule['pattern']!r}: {err}") from None
            checks.append((key, value))
        bounds = dict(checks)
        if "min" in bounds and "max" in bounds:
            lo, hi = bounds["min"], bounds["max"]
            if type(lo) is not type(hi):
                raise TypeError(f"{column!r}: min and max must both be numbers or both be dates")
            if lo > hi:
                raise ValueError(f"{column!r}: min ({_show(lo)}) is greater than max ({_show(hi)})")
        plans.append((column, checks))
    return plans


def _bound(column: Hashable, key: str, value: Any) -> float | pd.Timestamp:
    """A min/max value as a float, or as a Timestamp for dates (which may be written as text)."""
    if _is_number(value):
        if np.isnan(float(value)):
            raise ValueError(f"{column!r}: {key} can't be missing")
        return float(value)
    if isinstance(value, (str, dt.date, np.datetime64)):
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise ValueError(f"{column!r}: {key} must be a number or a date, got {value!r}") from None
        if pd.isna(stamp):
            raise ValueError(f"{column!r}: {key} can't be missing")
        return stamp
    raise TypeError(f"{column!r}: {key} must be a number or a date, got {value!r}")


# ---------- running one rule ----------

def _run(series: pd.Series, column: str, key: str, value: Any) -> RuleResult:
    if key == "required":
        missing = series.isna().to_numpy(dtype=bool)
        blank = np.zeros(len(series), dtype=bool)
        if _can_hold_text(series):
            filled = ~missing
            blank[filled] = _map(series[filled], lambda v: isinstance(v, str) and not v.strip())
        rows = np.flatnonzero(missing | blank)
        reasons = ["missing" if missing[p] else "empty text" for p in rows]
        return _result(series, column, key, "no blanks", len(series), rows, reasons)

    if key == "type" and value == "category":
        dtype = series.dtype
        note = "" if isinstance(dtype, pd.CategoricalDtype) else f"stored as {dtype}, not category"
        return RuleResult(column, key, "stored as category", 0, 0, note=note)

    present = np.flatnonzero(series.notna().to_numpy(dtype=bool))
    values = series.iloc[present]
    expected, check = _CHECKS[key](series, value)
    failed, reasons = check(values)
    rows = present[failed]
    return _result(series, column, key, expected, len(values), rows, reasons)


def _result(series: pd.Series, column: str, key: str, expected: str, checked: int,
            rows: np.ndarray, reasons: list[str]) -> RuleResult:
    values = [_plain(v) for v in series.iloc[rows]]
    return RuleResult(column, key, expected, checked, len(rows), rows=rows, values=values, reasons=reasons)


Check = Callable[[pd.Series], tuple[np.ndarray, list[str]]]


def _per_value(reason: Callable[[Any], str | None]) -> Check:
    """A check that looks at each value: reason() returns why it fails, or None."""
    def check(values: pd.Series) -> tuple[np.ndarray, list[str]]:
        found = [reason(v) for v in values]
        failed = np.array([r is not None for r in found], dtype=bool)
        return failed, [r for r in found if r is not None]
    return check


def _all_fail(reason: str) -> Check:
    def check(values: pd.Series) -> tuple[np.ndarray, list[str]]:
        return np.ones(len(values), dtype=bool), [reason] * len(values)
    return check


def _vector(mask: Callable[[pd.Series], Any], reason: str) -> Check:
    """A check done on the whole column at once; mask() is True where a value fails."""
    def check(values: pd.Series) -> tuple[np.ndarray, list[str]]:
        failed = np.asarray(mask(values), dtype=bool)
        return failed, [reason] * int(failed.sum())
    return check


def _type_check(series: pd.Series, want: str) -> tuple[str, Check]:
    dtype = series.dtype
    is_bool = pd.api.types.is_bool_dtype(dtype)
    is_number = pd.api.types.is_numeric_dtype(dtype) and not is_bool
    is_date = pd.api.types.is_datetime64_any_dtype(dtype)
    is_string = isinstance(dtype, pd.StringDtype)
    stored = "True/False" if is_bool else "a number" if is_number else "a date" if is_date else ""

    if want == "number":
        if is_number:
            return want, _vector(lambda v: np.zeros(len(v), dtype=bool), "")
        if stored:
            return want, _all_fail(f"{stored}, not a number")
    elif want == "integer":
        if pd.api.types.is_integer_dtype(dtype) and not is_bool:
            return want, _vector(lambda v: np.zeros(len(v), dtype=bool), "")
        if is_number:
            def not_whole(v: pd.Series) -> np.ndarray:
                x = v.to_numpy(dtype=float)
                with np.errstate(invalid="ignore"):   # infinity has no remainder
                    return ~(np.isfinite(x) & (np.mod(x, 1) == 0))
            return want, _vector(not_whole, "not a whole number")
        if stored:
            return want, _all_fail(f"{stored}, not a whole number")
    elif want == "text":
        if is_string:
            return want, _vector(lambda v: np.zeros(len(v), dtype=bool), "")
        if stored:
            return want, _all_fail(f"{stored}, not text")
    elif want == "boolean":
        if is_bool:
            return want, _vector(lambda v: np.zeros(len(v), dtype=bool), "")
        if is_date:
            return want, _all_fail("a date, not True/False")
    elif want == "datetime":
        if is_date:
            return want, _vector(lambda v: np.zeros(len(v), dtype=bool), "")
        if stored:
            return want, _all_fail(f"{stored}, not a date")
    return want, _per_value(lambda v: _type_reason(v, want))


def _type_reason(v: Any, want: str) -> str | None:
    if want == "number":
        if _is_number(v):
            return None
        if isinstance(v, str) and _looks_numeric(v):
            return "text that looks like a number (convert it with pd.to_numeric)"
        return f"{_describe(v)}, not a number"
    if want == "integer":
        if _is_number(v):
            return None if float(v).is_integer() else "not a whole number"
        return f"{_describe(v)}, not a whole number"
    if want == "text":
        return None if isinstance(v, str) else f"{_describe(v)}, not text"
    if want == "boolean":
        if isinstance(v, (bool, np.bool_)):
            return None
        if _is_number(v) and v in (0, 1):
            return "1/0, not True/False (dst.to_boolean converts it when you name the column)"
        if isinstance(v, str):
            return "text, not True/False (dst.to_boolean converts yes/no text)"
        return f"{_describe(v)}, not True/False"
    # datetime
    if _is_date(v):
        return None
    if isinstance(v, str):
        return "text, not a date (dst.to_datetime_columns converts date text)"
    return f"{_describe(v)}, not a date"


def _bound_check(key: str) -> Callable[[pd.Series, Any], tuple[str, Check]]:
    below = key == "min"

    def build(series: pd.Series, bound: float | pd.Timestamp) -> tuple[str, Check]:
        expected = f"{'≥' if below else '≤'} {_show(bound)}"
        word = "minimum" if below else "maximum"
        outside = f"{'below the' if below else 'above the'} {word} {_show(bound)}"
        dtype = series.dtype
        is_bool = pd.api.types.is_bool_dtype(dtype)
        is_number = pd.api.types.is_numeric_dtype(dtype) and not is_bool
        is_date = pd.api.types.is_datetime64_any_dtype(dtype)
        if isinstance(bound, pd.Timestamp):
            if is_number or is_bool:
                raise TypeError(f"{series.name!r}: {key} is a date, but the column holds "
                                f"{'True/False values' if is_bool else 'numbers'}")
            if is_date:
                try:
                    (series.iloc[:1] < bound) if below else (series.iloc[:1] > bound)
                except TypeError:
                    raise TypeError(f"{series.name!r}: {key} {_show(bound)} and the column's dates can't be "
                                    "compared; give both a time zone or neither") from None
                return expected, _vector(lambda v: (v < bound) if below else (v > bound), outside)
        else:
            if is_date:
                raise TypeError(f"{series.name!r}: {key} is a number, but the column holds dates")
            if is_bool:
                return expected, _all_fail("True/False, not a number")
            if is_number:
                return expected, _vector(lambda v: (v < bound) if below else (v > bound), outside)

        def reason(v: Any) -> str | None:
            if isinstance(bound, pd.Timestamp):
                if not _is_date(v):
                    return f"{_describe(v)}, not a date"
                try:
                    stamp = pd.Timestamp(v)
                    bad = stamp < bound if below else stamp > bound
                except TypeError:
                    return "can't compare dates with and without a time zone"
            else:
                if not _is_number(v):
                    return f"{_describe(v)}, not a number"
                bad = v < bound if below else v > bound
            return outside if bad else None
        return expected, _per_value(reason)
    return build


def _allowed_check(series: pd.Series, allowed: list[Any]) -> tuple[str, Check]:
    shown = [_plain(a) for a in allowed]
    expected = f"one of {shown!r}" if len(shown) <= 6 else f"one of {len(shown)} values"
    keys = set()
    for a in allowed:
        try:
            keys.add(_key(a))
        except TypeError:
            raise TypeError(f"{series.name!r}: allowed values must be simple values (text, numbers, "
                            f"dates, True/False), got {a!r}") from None
    lowered: dict[str, str] = {}
    for a in allowed:
        if isinstance(a, str):
            lowered.setdefault(a.casefold(), a)

    def reason(v: Any) -> str | None:
        try:
            if _key(v) in keys:
                return None
        except TypeError:
            return f"{_describe(v)}, not one of the allowed values"
        if isinstance(v, str) and v.strip().casefold() in lowered:
            return f"not allowed as written; matches {lowered[v.strip().casefold()]!r} ignoring case/spaces"
        if _is_number(v) and v in (0, 1) and any(isinstance(a, (bool, np.bool_)) for a in allowed):
            return "a number, not True/False"
        return "not one of the allowed values"
    return expected, _per_value(reason)


def _unique_check(series: pd.Series, _: Any) -> tuple[str, Check]:
    def check(values: pd.Series) -> tuple[np.ndarray, list[str]]:
        try:
            keys = [_key(v) for v in values]
        except TypeError:
            return np.ones(len(values), dtype=bool), ["value can't be compared for repeats"] * len(values)
        counts = Counter(keys)
        times = [counts[k] for k in keys]
        failed = np.array([n > 1 for n in times], dtype=bool)
        return failed, [f"repeated ({n} times)" for n in times if n > 1]
    return "no repeats", check


def _pattern_check(series: pd.Series, pattern: re.Pattern[str]) -> tuple[str, Check]:
    # Always Python's re, value by value: pandas 3 text columns may use a different
    # regex engine (RE2 via pyarrow), and the same pattern must mean the same thing everywhere.
    expected = f"matches {pattern.pattern}"

    def reason(v: Any) -> str | None:
        if not isinstance(v, str):
            if _is_number(v):
                return "a number, not text (numbers lose leading zeros)"
            return f"{_describe(v)}, not text"
        return None if pattern.fullmatch(v) else "doesn't match the pattern"
    return expected, _per_value(reason)


_CHECKS: dict[str, Callable[[pd.Series, Any], tuple[str, Check]]] = {
    "type": _type_check,
    "min": _bound_check("min"),
    "max": _bound_check("max"),
    "allowed": _allowed_check,
    "unique": _unique_check,
    "pattern": _pattern_check,
}


# ---------- helpers ----------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _is_date(v: Any) -> bool:
    return isinstance(v, (dt.date, np.datetime64))


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace(",", "").strip())
    except ValueError:
        return False
    return True


def _describe(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "True/False"
    if _is_number(v):
        return "a number"
    if isinstance(v, str):
        return "text"
    if _is_date(v):
        return "a date"
    return f"a {type(v).__name__}"


def _key(v: Any) -> tuple[str, Any]:
    """A comparison key that keeps True apart from 1 (Python treats True == 1)."""
    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v))
    if _is_number(v):
        return ("number", _plain(v))
    if _is_date(v):
        return ("date", pd.Timestamp(v))
    hash(v)
    return (type(v).__name__, v)


def _plain(v: Any) -> Any:
    """numpy scalars as plain Python values, so they print as 134 rather than np.int64(134)."""
    return v.item() if isinstance(v, np.generic) else v


def _show(v: Any) -> str:
    v = _plain(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, pd.Timestamp):
        return str(v.date()) if v == v.normalize() and v.tz is None else str(v)
    return repr(v)


def _can_hold_text(series: pd.Series) -> bool:
    dtype = series.dtype
    return (pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)
            or isinstance(dtype, pd.CategoricalDtype))


def _map(values: pd.Series, fn: Callable[[Any], bool]) -> np.ndarray:
    return np.array([fn(v) for v in values], dtype=bool)


def _escape(text: str) -> str:
    return html.escape(text)
