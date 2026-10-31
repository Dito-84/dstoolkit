"""One-call dataset report: overview, per-column profiles, correlations and
suggested cleaning code.

    import dstoolkit as dst

    report = dst.profile(df, title="Titanic passengers")
    report                      # shows inline in Jupyter
    report.save("report.html")  # one self-contained file to share
    report.suggestions          # the recommended cleaning steps as data
    print(report.code())        # ...or as ready-to-run dstoolkit code

    # add a validation section: rules and checks as in dst.validate()
    report = dst.profile(df, rules={"age": {"min": 0, "max": 120}})
    report.validation.ok

    report.associations         # related columns of any type (see dst.associations)

    # preparing for a model: name what it will predict
    report = dst.profile(df, target="survived")
    report.target.leaks         # columns that may give the answer away
    print(report.pipeline())    # a leak-free scikit-learn starter, ready to run

The report never changes the DataFrame it describes.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .cleaning import (
    DATE_PATTERN,
    DOB_NAME_HINTS,
    _bounds,
    _date_order,
    _is_yes_no,
    _normalized,
    _parse_dates,
    _standard_spellings,
)
from .exploration import associations as _associations
from .validation import ColumnRule, Comparison, ValidationReport, validate

__all__ = ["profile", "Profile", "ColumnProfile", "Suggestion", "TargetProfile"]

Kind = Literal["numeric", "categorical", "boolean", "datetime", "text", "empty"]
Severity = Literal["high", "medium", "low"]
Task = Literal["classification", "regression"]

MOSTLY_EMPTY = 0.60          # share missing above which a column is suggested for dropping
OUTLIER_SHARE_TO_CAP = 0.01  # share of outliers above which capping is suggested
LONG_TAIL_SHARE = 0.10       # above this share, "outliers" are a long tail: suggest log, not capping
SKEW_FOR_LOG = 1.0           # skewness above which a log transform is suggested
DISCRETE_MAX_UNIQUE = 10     # numeric columns with few values (counts, codes) skip outlier advice
TEXT_MIN_UNIQUE = 50         # text columns with more distinct values than this...
TEXT_MIN_RATIO = 0.2         # ...and this share of distinct values are treated as free text / IDs
HIGH_CORRELATION = 0.9
RELATED_MAX_COLUMNS = 30     # above this, the related-columns section is skipped (pairs grow with the square)
RELATED_ROWS_SHOWN = 20      # pairs listed in the HTML; report.associations has them all
LEAK_STRENGTH = 0.9          # a feature this strongly related to the target may give the answer away
MISSING_SIGNAL = 0.1         # blanks related to the target at least this strongly are worth a flag
IMBALANCE_SHARE = 0.10       # smallest class below this share: the target is imbalanced
CLASSIFICATION_MAX_VALUES = 10  # a numeric target with at most this many whole-number values is a class label

# Order suggested steps are applied in, matching dst.clean()
STEP_ORDER = {"duplicates": 0, "drop": 1, "type": 2, "missing": 3, "outliers": 4, "transform": 5,
              "encode": 6, "derive": 7, "note": 8}

COLORS = {"bar": "#4c7cc4", "accent": "#c2410c", "grid": "#d9dee5", "ink": "#3d4653"}


@dataclass
class Suggestion:
    """One recommended cleaning step.

    action/options describe the step as code (see Profile.code()); notes that
    need a judgment call have no action.
    """
    column: str | None
    step: str
    severity: Severity
    message: str
    action: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnProfile:
    name: str
    kind: Kind
    dtype: str
    count: int
    missing: int
    unique: int
    stats: dict[str, Any] = field(default_factory=dict)
    top_values: list[tuple[str, int]] = field(default_factory=list)
    outliers: int | None = None
    chart: str | None = None  # base64 PNG
    missing_expected: bool = False  # blanks carry meaning (see profile(missing_ok=...))

    @property
    def missing_share(self) -> float:
        total = self.count + self.missing
        return self.missing / total if total else 0.0

    @property
    def missing_label(self) -> str:
        label = f"{100 * self.missing_share:.1f}%"
        return f"{label} (expected)" if self.missing_expected and self.missing else label


@dataclass
class TargetProfile:
    """What the report found about the column a model will predict (profile(target=...))."""
    name: str
    task: Task
    missing: int
    classes: pd.Series | None      # rows per class (classification only), largest first
    features: pd.DataFrame         # every testable feature vs the target, strongest first
    missingness: pd.DataFrame      # features whose blanks are related to the target
    leaks: list[str]               # features that may give the answer away
    untested: list[str]            # features associations() couldn't test (IDs, dates, ...)


class Profile:
    """A dataset report. Display it in Jupyter, save it as HTML, or read its parts."""

    def __init__(self, title: str, n_rows: int, n_columns: int, duplicates: int, missing_cells: int,
                 memory_bytes: int, columns: list[ColumnProfile], correlations: pd.DataFrame | None,
                 correlation_chart: str | None, suggestions: list[Suggestion],
                 validation: ValidationReport | None = None,
                 associations: pd.DataFrame | None = None,
                 associations_skipped: str | None = None,
                 target: TargetProfile | None = None) -> None:
        self.title = title
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.duplicates = duplicates
        self.missing_cells = missing_cells
        self.memory_bytes = memory_bytes
        self.columns = columns
        self.correlations = correlations
        self.correlation_chart = correlation_chart
        self.suggestions = suggestions
        self.validation = validation
        self.associations = associations
        self.associations_skipped = associations_skipped
        self.target = target

    # ----- data views -----

    def summary(self) -> pd.DataFrame:
        """One row per column: type, missing, unique values and outliers."""
        return pd.DataFrame([{
            "column": c.name, "type": c.kind, "dtype": c.dtype, "non_missing": c.count,
            "missing": c.missing, "missing_%": round(100 * c.missing_share, 1),
            "missing_expected": c.missing_expected,
            "unique": c.unique, "outliers": c.outliers,
        } for c in self.columns]).set_index("column")

    def code(self, df_name: str = "df", suggestions: Iterable[Suggestion] | None = None) -> str:
        """The suggested cleaning steps as runnable Python, one call per kind of step.

        suggestions: only include these (e.g. the ones a user ticked); default all.
        """
        chosen = self.suggestions if suggestions is None else list(suggestions)
        lines = self._code_lines(df_name, chosen)
        header = ["import pandas as pd"] if any("pd." in line for line in lines) else []
        header += ["import dstoolkit as dst", ""]
        return "\n".join(header + (lines or ["# No cleaning steps suggested"]))

    def _code_lines(self, d: str, chosen: list[Suggestion]) -> list[str]:
        groups: dict[tuple[str, tuple[tuple[str, Any], ...]], list[Suggestion]] = {}
        for s in chosen:
            if s.action:
                groups.setdefault((s.action, tuple(sorted(s.options.items()))), []).append(s)

        def cols(items: list[Suggestion]) -> str:
            return "[" + ", ".join(repr(i.column) for i in items) + "]"

        lines: list[str] = []
        for (action, opts), items in groups.items():
            options = dict(opts)
            if action == "remove_duplicates":
                lines.append(f"{d} = dst.remove_duplicates({d})")
            elif action == "drop":
                lines.append(f"{d} = {d}.drop(columns={cols(items)})")
            elif action == "to_boolean":
                lines.append(f"{d} = dst.to_boolean({d}, {cols(items)})")
            elif action == "to_datetime":
                lines.append(f"{d} = dst.to_datetime_columns({d}, {cols(items)})")
            elif action == "to_numeric":
                lines += [f"{d}[{i.column!r}] = pd.to_numeric({d}[{i.column!r}].astype(str)"
                          f".str.replace(',', '', regex=False).str.strip(), errors='coerce')" for i in items]
            elif action == "fill_missing":
                key, method = next(iter(options.items()))
                lines.append(f"{d} = dst.fill_missing({d}, {key}={method!r}, columns={cols(items)})")
            elif action == "cap_outliers":
                lines.append(f"{d} = dst.handle_outliers({d}, {cols(items)}, action='cap')")
            elif action == "log_transform":
                lines.append(f"{d} = dst.log_transform({d}, {cols(items)}, suffix='_log')")
            elif action == "encode":
                mapping = ", ".join(f"{i.column!r}: {i.options['method']!r}" for i in items)
                lines.append(f"{d} = dst.encode_categorical({d}, {{{mapping}}})")
            elif action == "standardize":
                lines += [f"{d} = dst.standardize_labels({d}, {i.column!r}, {list(i.options['labels'])!r})"
                          for i in items]
            elif action == "add_age":
                lines += [f"{d} = dst.add_age_column({d}, dob_column={i.column!r})" for i in items]
            elif action == "drop_missing_target":
                lines.append(f"{d} = {d}.dropna(subset=[{items[0].column!r}])")

        # one encode call is easier to read than one per method
        encodes = [line for line in lines if "dst.encode_categorical(" in line]
        if len(encodes) > 1:
            merged = ", ".join(line.split("{", 1)[1].rsplit("}", 1)[0] for line in encodes)
            first = lines.index(encodes[0])
            lines = [line for line in lines if line not in encodes]
            lines.insert(first, f"{d} = dst.encode_categorical({d}, {{{merged}}})")

        # type conversions first, so later steps see True/False and dates rather than text
        conversions = ("dst.to_boolean(", "dst.to_datetime_columns(", "pd.to_numeric(", "dst.standardize_labels(")
        return [line for line in lines if any(c in line for c in conversions)] + \
               [line for line in lines if not any(c in line for c in conversions)]

    def pipeline(self, df_name: str = "df") -> str:
        """A leak-free modelling starter as runnable Python: needs profile(target=...).

        Steps that learn nothing from the data (type fixes, log transforms,
        dropping rows without a target) run before the split. Steps that learn
        from it (fill values, outlier bounds, categories) go in a scikit-learn
        Pipeline of dstoolkit transformers, so cross-validation refits them on
        each training fold. Possible leaks, identifiers, dates and columns
        suggested for dropping are left out of the features, with a comment
        saying why. The model is a histogram gradient-boosting model, which
        accepts the remaining blanks and needs no scaling. A no-skill baseline is
        scored on the same folds, so the result has something to beat.
        """
        t = self.target
        if t is None:
            raise ValueError("pipeline() needs to know what the model predicts: profile(df, target='column')")
        d = df_name
        kinds = {c.name: c.kind for c in self.columns}
        by_action: dict[str, list[Suggestion]] = {}
        for s in self.suggestions:
            if s.action and s.column != t.name:
                by_action.setdefault(s.action, []).append(s)

        def cols(action: str) -> list[str]:
            return [s.column for s in by_action.get(action, []) if s.column is not None]

        dropped, dates = cols("drop"), [n for n, k in kinds.items() if k == "datetime"]
        identifiers = [n for n, k in kinds.items() if k == "text" and n not in cols("to_numeric")]
        converted = set(cols("to_numeric") + cols("to_boolean") + cols("standardize"))
        encoded = set(cols("encode"))
        unencoded = [n for n, k in kinds.items() if k == "categorical" and n not in encoded
                     and n not in converted and n != t.name]
        excluded = {t.name, *dropped, *dates, *identifiers, *t.leaks, *unencoded}
        features = [n for n in kinds if n not in excluded]
        booleans = [n for n in features if kinds[n] == "boolean" or n in cols("to_boolean")]

        before = self._code_lines(d, [s for s in self.suggestions if s.action in
                                      ("remove_duplicates", "to_numeric", "to_boolean", "standardize")
                                      and s.column not in excluded])
        if t.missing:
            before.append(f"{d} = {d}.dropna(subset=[{t.name!r}])   # rows without a target can't be used")
        logs = [c for c in cols("log_transform") if c in features]
        if logs:
            before.append(f"{d} = dst.log_transform({d}, {logs})   # log(1 + x) learns nothing from the data")
        if booleans:
            before.append(f"{d}[{booleans}] = {d}[{booleans}].astype('boolean').astype(float)   "
                          f"# True/False as 1/0; blanks stay missing")

        steps: list[str] = []
        fills: dict[tuple[str, str], list[str]] = {}
        for s in by_action.get("fill_missing", []):
            if s.column in features and s.column not in booleans:
                key, method = next(iter(s.options.items()))
                fills.setdefault((key, method), []).append(str(s.column))
        for (key, method), names in fills.items():
            other = "categorical" if key == "numeric" else "numeric"
            steps.append(f"dst.MissingFiller({key}={method!r}, {other}=None, columns={names}),")
        capped = [c for c in cols("cap_outliers") if c in features]
        if capped:
            steps.append(f"dst.OutlierCapper(columns={capped}),")
        encodings = {str(s.column): s.options["method"] for s in by_action.get("encode", []) if s.column in features}
        if encodings:
            steps.append(f"dst.CategoryEncoder({encodings!r}, drop_original=True),")
        classify = t.task == "classification"
        model = "HistGradientBoostingClassifier" if classify else "HistGradientBoostingRegressor"
        dummy = "DummyClassifier" if classify else "DummyRegressor"
        guess = "the most common class" if classify else "the average"
        steps.append(f"{model}(random_state=0),   # accepts blanks and needs no scaling")

        if not classify:
            folds, scoring = "KFold(5, shuffle=True, random_state=0)", "neg_root_mean_squared_error"
        else:
            folds = "StratifiedKFold(5, shuffle=True, random_state=0)   # keeps each class's share in every fold"
            scoring = "roc_auc" if t.classes is not None and len(t.classes) == 2 else "balanced_accuracy"
        left_out = [("possible leaks: check they were known before the outcome", t.leaks),
                    ("suggested for dropping", [c for c in dropped if c != t.name and c not in t.leaks]),
                    ("identifiers or free text", identifiers),
                    ("dates: turn into features first if they matter", dates),
                    ("categories with no encoding suggested", unencoded)]
        lines = [*(["import pandas as pd"] if any("pd." in line for line in before) else []),
                 "import dstoolkit as dst",
                 f"from sklearn.dummy import {dummy}",
                 f"from sklearn.ensemble import {model}",
                 f"from sklearn.model_selection import {'StratifiedKFold' if classify else 'KFold'}, cross_val_score",
                 "from sklearn.pipeline import make_pipeline", "",
                 "# 1. Steps that learn nothing from the data: safe before the split", *before]
        if not before:
            lines.append("# (none needed)")
        flags = [str(c) for c in t.missingness["column_2"] if c not in t.leaks]
        if flags:
            lines.append("# Blanks related to the target: add a flag only if the blank was known before the outcome")
            lines += [f"# {d}[{f'{c}_missing'!r}] = {d}[{c!r}].isna().astype(int)" for c in flags]
        lines += ["", "# 2. Features and target"]
        listed: set[str] = set()
        for why, names in left_out:
            fresh = [n for n in names if n not in listed]
            listed.update(fresh)
            if fresh:
                lines.append(f"# Left out ({why}): {', '.join(fresh)}")
        lines += [f"X = {d}[{features!r}]", f"y = {d}[{t.name!r}]", "",
                  "# 3. Steps that learn from the data: inside the pipeline, refitted on each training fold",
                  "# To use a model that can't handle blanks (e.g. LogisticRegression): fill every column,",
                  "# and add dst.Scaler(...) before a linear or distance-based model",
                  "model = make_pipeline(", *[f"    {step}" for step in steps], ")", "",
                  f"folds = {folds}",
                  f"scores = cross_val_score(model, X, y, cv=folds, scoring={scoring!r})",
                  f"baseline = cross_val_score({dummy}(), X, y, cv=folds, scoring={scoring!r})   "
                  f"# always predicts {guess}",
                  f'print(f"model:    {scoring} {{scores.mean():.3f}} ± {{scores.std():.3f}}")',
                  f'print(f"baseline: {scoring} {{baseline.mean():.3f}}   (no skill: the score to beat)")']
        return "\n".join(lines)

    # ----- HTML -----

    def _fragment(self) -> str:
        e = html.escape
        missing_pct = 100 * self.missing_cells / max(1, self.n_rows * self.n_columns)
        tiles = [
            ("Rows", f"{self.n_rows:,}"),
            ("Columns", f"{self.n_columns:,}"),
            ("Duplicate rows", f"{self.duplicates:,}"),
            ("Missing cells", f"{self.missing_cells:,} ({missing_pct:.1f}%)"),
            ("Memory", _human_bytes(self.memory_bytes)),
        ]
        v = self.validation
        if v is not None:
            tiles.append(("Rules failed", f"{len(v.failed_rules)} of {len(v.results)}"))
        parts = [f'<div class="dst-profile"><h1>{e(self.title)}</h1>',
                 '<div class="dst-tiles">' + "".join(
                     f'<div class="dst-tile"><div class="dst-label">{e(k)}</div>'
                     f'<div class="dst-value">{e(v)}</div></div>'
                     for k, v in tiles) + "</div>"]

        if v is not None:
            parts.append(_validation_html(v))

        if self.target is not None:
            parts.append(self._target_html())

        # Suggestions
        parts.append("<h2>Suggested cleaning</h2>")
        if self.suggestions:
            parts.append('<ul class="dst-suggestions">' + "".join(
                f'<li class="dst-{s.severity}"><span class="dst-badge">{s.severity}</span>'
                f'{"<code>" + e(s.column) + "</code>: " if s.column else ""}{e(s.message)}</li>'
                for s in self.suggestions) + "</ul>")
            parts.append('<p class="dst-note">Ready-to-run code for these steps (review before using):</p>'
                         f'<pre class="dst-code">{e(self.code())}</pre>')
        else:
            parts.append('<p class="dst-note">No issues found.</p>')

        # Overview table
        parts.append("<h2>Columns</h2><div class=\"dst-scroll\"><table class=\"dst-table\"><thead><tr>"
                     "<th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Summary</th></tr></thead><tbody>")
        for c in self.columns:
            bar = f'<span class="dst-meter"><span style="width:{100 * c.missing_share:.0f}%"></span></span>'
            parts.append(f'<tr><td><a href="#dst-col-{_anchor(c.name)}">{e(c.name)}</a></td><td>{e(c.kind)}</td>'
                         f'<td>{bar} {e(c.missing_label)}</td><td>{c.unique:,}</td>'
                         f'<td>{e(_one_line(c))}</td></tr>')
        parts.append("</tbody></table></div>")

        # Column cards
        parts.append('<h2>Column details</h2><div class="dst-cards">')
        for c in self.columns:
            stats = "".join(f"<tr><th>{e(k)}</th><td>{e(_fmt(v))}</td></tr>" for k, v in c.stats.items())
            notes = [s for s in self.suggestions if s.column == c.name]
            note_html = "".join(f'<li class="dst-{s.severity}">{e(s.message)}</li>' for s in notes)
            if v is not None:
                note_html = "".join(
                    f'<li class="dst-high">Failed rule: {e(_rule_label(r))} ({r.failed:,} row(s))</li>'
                    for r in v.failed_rules if c.name in r.columns and not r.note) + note_html
            img = (f'<img alt="Distribution of {e(c.name)}" src="data:image/png;base64,{c.chart}">'
                   if c.chart else "")
            parts.append(
                f'<section class="dst-card" id="dst-col-{_anchor(c.name)}"><h3>{e(c.name)} '
                f'<span class="dst-kind">{e(c.kind)} · {e(c.dtype)}</span></h3>{img}'
                f'<table class="dst-stats">{stats}</table>'
                f'{"<ul class=dst-col-notes>" + note_html + "</ul>" if note_html else ""}</section>')
        parts.append("</div>")

        # Correlations
        if self.correlations is not None and not self.correlations.empty:
            parts.append("<h2>Correlations</h2>"
                         '<p class="dst-note">Pearson correlation between numeric columns (−1 to 1). '
                         'Blank cells are undefined: one column doesn\'t vary where both have values.</p>')
            if self.correlation_chart:
                parts.append(f'<img class="dst-heatmap" alt="Correlation heatmap" '
                             f'src="data:image/png;base64,{self.correlation_chart}">')
            pairs = _strong_pairs(self.correlations, 0.5)
            if pairs:
                parts.append('<h3>Notable pairs (|r| ≥ 0.5)</h3>'
                             '<table class="dst-table"><thead><tr><th>Column</th><th>Column</th><th>r</th></tr>'
                             '</thead><tbody>'
                             + "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{r:+.2f}</td></tr>" for a, b, r in pairs)
                             + "</tbody></table>")
        parts.append(_related_html(self.associations, self.associations_skipped))
        parts.append('<p class="dst-footer">Generated by dstoolkit.profile()</p></div>')
        return "\n".join(parts)

    def _target_html(self) -> str:
        e = html.escape
        t = self.target
        assert t is not None
        parts = [f"<h2>Target: {e(t.name)}</h2>",
                 f'<p class="dst-note">Treated as <b>{t.task}</b> (pass <code>task=</code> to change it). '
                 + (f"{t.missing:,} row(s) have no target and can't be used for training." if t.missing else
                    "Every row has a target.") + "</p>"]
        if t.classes is not None:
            total = int(t.classes.sum())
            parts.append('<table class="dst-table"><thead><tr><th>Class</th><th>Rows</th><th>Share</th></tr></thead>'
                         "<tbody>" + "".join(f"<tr><td>{e(str(k))}</td><td>{int(v):,}</td><td>{v / total:.1%}</td></tr>"
                                             for k, v in t.classes.head(20).items()) + "</tbody></table>")
        leaks = set(t.leaks)
        if not t.features.empty:
            rows = "".join(
                f"<tr><td>{e(str(r['column_2']))}</td><td>{e(str(r['test']))}</td>"
                f"<td>{e(_strength_text(str(r['measure']), float(r['strength'])))}</td>"
                f"<td>{_p_text(float(r['p_adjusted']))}</td><td>{int(r['n']):,}</td>"
                f"<td>{'<b>possible leak</b> ' if str(r['column_2']) in leaks else ''}{e(str(r['note']))}</td></tr>"
                for r in t.features.to_dict("records"))
            parts.append("<h3>How strongly each feature is related to the target</h3>"
                         '<p class="dst-note">Tested on the rows where both values are present; p-values are '
                         "adjusted for testing many features. A strong relation can be real signal, or a leak: "
                         "a column recorded after the outcome. Related isn't the same as useful in a model, "
                         "where features overlap.</p>"
                         '<div class="dst-scroll"><table class="dst-table"><thead><tr><th>Feature</th><th>Test</th>'
                         "<th>Strength</th><th>Adjusted p</th><th>Rows</th><th>Note</th></tr></thead><tbody>"
                         + rows + "</tbody></table></div>")
        if t.untested:
            parts.append(f'<p class="dst-note">Not tested (identifiers, free text, dates, or one value where the '
                         f"target is present): {e(', '.join(t.untested))}.</p>")
        if not t.missingness.empty:
            rows = "".join(
                f"<tr><td>{e(str(r['column_2']))}</td>"
                f"<td>{e(_strength_text(str(r['measure']), float(r['strength'])))}</td>"
                f"<td>{_LEAK if str(r['column_2']) in leaks else 'keep a flag if known before the outcome'}</td></tr>"
                for r in t.missingness.to_dict("records"))
            parts.append("<h3>Blanks related to the target</h3>"
                         '<p class="dst-note">Whether the value is blank, compared with the target. Filling '
                         "these blanks hides that information unless a flag column keeps it.</p>"
                         '<table class="dst-table"><thead><tr><th>Feature</th><th>Strength</th><th>Meaning</th>'
                         "</tr></thead><tbody>" + rows + "</tbody></table>")
        parts.append("<h3>Leak-free modelling starter</h3>"
                     '<p class="dst-note">From <code>report.pipeline()</code>: the learned steps run inside a '
                     "scikit-learn pipeline, so cross-validation refits them on each training fold. Review it "
                     "before using.</p>"
                     f'<pre class="dst-code">{e(self.pipeline())}</pre>')
        return "\n".join(parts)

    def to_html(self) -> str:
        """A complete, self-contained HTML page."""
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                f"<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>"
                f'<body class="dst-body">{self._fragment()}</body></html>')

    def save(self, path: str | Path) -> Path:
        """Write the report to an HTML file and return its path."""
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return f"<style>{_CSS}</style>{self._fragment()}"

    def __repr__(self) -> str:
        return (f"<Profile {self.title!r}: {self.n_rows:,} rows × {self.n_columns} columns, "
                f"{len(self.suggestions)} suggestions>")


# ---------- building the profile ----------

def profile(df: pd.DataFrame, title: str = "Dataset profile", charts: bool = True,
            max_categories: int = 10, missing_ok: Iterable[str] = (),
            rules: Mapping[Hashable, ColumnRule] | None = None,
            checks: Mapping[str, Comparison] | None = None, related: bool = True,
            target: Hashable | None = None, task: Literal["auto", "classification", "regression"] = "auto",
            ) -> Profile:
    """Profile a DataFrame: overview, per-column statistics, correlations and
    suggested cleaning steps.

    charts: draw a small chart for each column and a correlation heatmap
    max_categories: how many of the most common values to show per text column
    missing_ok: columns where a blank means something (e.g. "no complication",
                "not readmitted"), so they get no suggestions to fill or drop them
    rules, checks: validation rules, as in `dst.validate()`; when given, the
                   report gets a Validation section and `report.validation`
    related: add a "Related columns" section and `report.associations`: pairs
             of columns of any type that are related, from `dst.associations()`
             with its defaults. Skipped above 30 columns, where the number of
             pairs makes it slow; call `dst.associations()` on the columns you
             care about instead.
    target: the column a model will predict. Adds a Target section: the class
            balance, how strongly each feature is related to the target,
            features whose blanks are related to it, and possible leaks
            (features almost perfectly related to it). The target itself gets
            no fill, encoding or outlier advice, and `report.pipeline()` writes
            a leak-free scikit-learn starter.
    task: "classification" or "regression"; "auto" treats text, True/False and
          whole-number columns with at most 10 values as classes
    """
    if df.shape[1] == 0:
        raise ValueError("DataFrame has no columns")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Column name(s) {dupes} appear more than once; rename duplicates first")

    if target is not None and target not in df.columns:
        raise KeyError(f"target {target!r} is not a column")
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f'task must be "auto", "classification" or "regression", got {task!r}')
    validation = (validate(df, rules, checks, verbose=False)
                  if rules is not None or checks is not None else None)

    expected = {str(c) for c in missing_ok}
    unknown = sorted(expected - {str(c) for c in df.columns})
    if unknown:
        raise KeyError(f"missing_ok names columns that don't exist: {unknown}")

    columns: list[ColumnProfile] = []
    suggestions: list[Suggestion] = []
    duplicates = int(df.duplicated().sum())
    if duplicates:
        suggestions.append(Suggestion(None, "duplicates", "high",
                                      f"{duplicates:,} duplicate row(s).", "remove_duplicates"))

    names = [str(c) for c in df.columns]
    for name in df.columns:
        col_profile, col_suggestions = _profile_column(df[name], str(name), charts, max_categories,
                                                       str(name) in expected)
        columns.append(col_profile)
        suggestions.extend(s for s in col_suggestions if not _already_done(s, names))

    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    correlations = numeric.corr() if numeric.shape[1] >= 2 else None
    corr_chart = None
    if correlations is not None:
        for a, b, r in _strong_pairs(correlations, HIGH_CORRELATION):
            suggestions.append(Suggestion(b, "note", "low",
                                          f"Almost perfectly correlated with {a} (r = {r:+.2f}); "
                                          f"one of them may be redundant."))
        if charts and correlations.shape[0] <= 25:
            corr_chart = _heatmap(correlations)

    related_pairs: pd.DataFrame | None = None
    related_skipped: str | None = None
    if not related:
        related_skipped = "turned off (related=False)"
    elif df.shape[1] > RELATED_MAX_COLUMNS:
        related_skipped = (f"skipped: {df.shape[1]} columns is more than {RELATED_MAX_COLUMNS}. "
                           "Run dst.associations(df, columns=[...]) on the columns you're interested in")
    else:
        related_pairs = _associations(df, verbose=False)

    target_profile = None
    if target is not None:
        target_profile, target_suggestions = _profile_target(df, target, task)
        suggestions = [s for s in suggestions if s.column != target_profile.name or s.step == "type"]
        suggestions.extend(target_suggestions)

    order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: (STEP_ORDER.get(s.step, 9), order[s.severity]))

    return Profile(title=title, n_rows=len(df), n_columns=df.shape[1], duplicates=duplicates,
                   missing_cells=int(df.isna().sum().sum()),
                   memory_bytes=int(df.memory_usage(deep=True).sum()), columns=columns,
                   correlations=correlations, correlation_chart=corr_chart, suggestions=suggestions,
                   validation=validation, associations=related_pairs, associations_skipped=related_skipped,
                   target=target_profile)


def _task(series: pd.Series, task: str) -> Task:
    if task in ("classification", "regression"):
        return task  # type: ignore[return-value]
    values = series.dropna()
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "classification"
    whole = bool((values.astype(float) % 1 == 0).all())
    return "classification" if whole and values.nunique() <= CLASSIFICATION_MAX_VALUES else "regression"


def _profile_target(df: pd.DataFrame, target: Hashable,
                    task: str) -> tuple[TargetProfile, list[Suggestion]]:
    name = str(target)
    series = df[target]
    kind = _task(series, task)
    out: list[Suggestion] = []
    missing = int(series.isna().sum())
    if missing:
        out.append(Suggestion(name, "drop", "high", f"{missing:,} row(s) have no {name}: drop them before "
                              f"modelling (a target can't be filled).", "drop_missing_target"))

    classes = None
    data = df.copy()
    if kind == "classification":
        classes = series.value_counts()
        data[target] = series.astype("category")   # class labels, even when stored as 0/1
        if len(classes) < 2:
            out.append(Suggestion(name, "note", "high", f"{name} has {len(classes)} value(s): nothing to predict."))
        elif classes.iloc[-1] / classes.sum() < IMBALANCE_SHARE:
            out.append(Suggestion(name, "note", "medium", f"Imbalanced: the smallest class ({classes.index[-1]!s}) "
                                  f"is {classes.iloc[-1] / classes.sum():.1%} of rows. Use stratified folds and a "
                                  f"metric such as ROC AUC or balanced accuracy, not plain accuracy."))

    empty = pd.DataFrame(columns=["column_1", "column_2", "measure", "strength", "p_adjusted", "n"])
    untested: list[str] = []
    features = empty
    if series.nunique() >= 2:
        features = _associations(data, target=target, significant_only=False, verbose=False)
        tested = {str(c) for c in features["column_2"]}
        untested = [str(c) for c in df.columns if c != target and str(c) not in tested]
        features = features.assign(column_1=features["column_1"].map(str), column_2=features["column_2"].map(str))

    gaps = [c for c in df.columns if c != target and 0 < df[c].isna().sum() < len(df)]
    missingness = empty
    if gaps and series.nunique() >= 2:
        blanks = pd.DataFrame({str(c): df[c].isna() for c in gaps}, index=df.index)
        blanks[name] = data[target]
        found = _associations(blanks, target=name, significant_only=False, verbose=False)
        missingness = found[(found["p_adjusted"] < 0.05) & (found["strength"] >= MISSING_SIGNAL)]
        missingness = missingness.reset_index(drop=True)

    leaks: list[str] = []
    strong = features[(features["p_adjusted"] < 0.05) & (features["strength"].abs() >= LEAK_STRENGTH)]
    for r in strong.to_dict("records"):
        leaks.append(str(r["column_2"]))
        out.append(Suggestion(str(r["column_2"]), "drop", "high",
                              f"Possible leak: almost perfectly related to {name} ({r['measure']} = "
                              f"{abs(r['strength']):.2f}). Check it was known before the outcome; if not, "
                              f"leave it out of the model."))
    for r in missingness.to_dict("records"):
        column = str(r["column_2"])
        if r["strength"] >= LEAK_STRENGTH:
            if column not in leaks:
                leaks.append(column)
            out.append(Suggestion(column, "drop", "high",
                                  f"Possible leak: whether {column} is blank predicts {name} almost perfectly "
                                  f"({r['measure']} = {r['strength']:.2f}). It may have been recorded after the "
                                  f"outcome; if so, leave it out of the model."))
        else:
            out.append(Suggestion(column, "note", "medium",
                                  f"Whether {column} is blank is related to {name} ({r['measure']} = "
                                  f"{r['strength']:.2f}). If blanks were known before the outcome, keep a flag "
                                  f"before filling: df[{f'{column}_missing'!r}] = df[{column!r}].isna()."))
    return TargetProfile(name=name, task=kind, missing=missing, classes=classes, features=features,
                         missingness=missingness, leaks=leaks, untested=untested), out


def _rule_label(r: Any) -> str:
    return f"{r.column}: {r.expected}" if r.rule == "compare" else f"{r.rule} ({r.expected})"


def _validation_html(v: ValidationReport) -> str:
    e = html.escape
    headline = v.message().splitlines()[0]
    rows = []
    for r in sorted(v.results, key=lambda r: r.passed):
        status = '<span class="dst-pass">passed</span>' if r.passed else '<span class="dst-fail">failed</span>'
        name = r.column if r.rule != "compare" else f"{r.column} (check)"
        detail = r.note or v._examples(r, v.max_examples)
        rows.append(f"<tr><td>{status}</td><td>{e(name)}</td><td>{e(r.rule)}</td><td>{e(r.expected)}</td>"
                    f"<td>{r.failed:,} of {r.checked:,}</td><td>{e(detail)}</td></tr>")
    return ("<h2>Validation</h2>"
            f'<p class="dst-note">{e(headline)} Failed rules come first; '
            "<code>report.validation.failures()</code> lists every failing value.</p>"
            '<div class="dst-scroll"><table class="dst-table"><thead><tr><th>Result</th><th>Column</th>'
            "<th>Rule</th><th>Expected</th><th>Failed</th><th>Examples</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def _kind(series: pd.Series) -> Kind:
    values = series.dropna()
    if values.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    text = values.astype(str)
    sample = text.sample(min(len(text), 500), random_state=0)
    if sample.str.match(DATE_PATTERN).mean() >= 0.9:
        return "datetime"
    unique = values.nunique()
    if unique > TEXT_MIN_UNIQUE and unique / len(values) > TEXT_MIN_RATIO:
        return "text"
    return "categorical"


ID_NAME_HINT = re.compile(r"(^|[_\s])(id|code|zip|zipcode|postal|phone|ccn|npi|mrn|ssn|account)($|[_\s])|id$",
                          re.IGNORECASE)


def _looks_like_identifier(series: pd.Series) -> bool:
    """Digit strings that are really labels (IDs, ZIP codes): converting them to numbers loses leading zeros."""
    if ID_NAME_HINT.search(str(series.name or "")):
        return True
    text = series.dropna().astype(str).str.strip()
    return bool(text.str.match(r"^0\d").any())


def _numbers_stored_as_text(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty or pd.api.types.is_numeric_dtype(series) or _looks_like_identifier(series):
        return False
    converted = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return bool(converted.notna().mean() >= 0.95)


def _profile_column(series: pd.Series, name: str, charts: bool, max_categories: int,
                    missing_expected: bool = False) -> tuple[ColumnProfile, list[Suggestion]]:
    kind = _kind(series)
    values = series.dropna()
    missing = int(series.isna().sum())
    # When blanks are meaningful, profile the values but never suggest filling or dropping
    share = 0.0 if missing_expected else (missing / len(series) if len(series) else 0.0)
    col = ColumnProfile(name=name, kind=kind, dtype=str(series.dtype), count=int(values.size),
                        missing=missing, unique=int(values.nunique()), missing_expected=missing_expected)
    out: list[Suggestion] = []

    def suggest(step: str, severity: Severity, message: str, action: str | None = None,
                **options: Any) -> None:
        out.append(Suggestion(name, step, severity, message, action, options))

    if kind == "empty":
        if not missing_expected:
            suggest("drop", "high", "Every value is missing; drop this column.", "drop")
        return col, out

    numeric_text = _numbers_stored_as_text(series)
    if numeric_text:
        suggest("type", "high", "Numbers stored as text; convert to a numeric type.", "to_numeric")

    if kind == "numeric":
        x = values.astype(float)
        raw_skew = x.skew() if len(x) > 2 else 0.0
        skew = float(raw_skew) if isinstance(raw_skew, (int, float, np.number)) else 0.0
        skew = 0.0 if np.isnan(skew) else skew
        col.stats = {"mean": x.mean(), "std": x.std(), "min": x.min(), "25%": x.quantile(0.25),
                     "median": x.median(), "75%": x.quantile(0.75), "max": x.max(), "skew": skew,
                     "zeros": int((x == 0).sum())}
        discrete = col.unique <= DISCRETE_MAX_UNIQUE
        if not discrete:
            low, high = _bounds(x, "iqr", None)
            col.outliers = int(((x < low) | (x > high)).sum())
            col.stats["outliers (IQR)"] = col.outliers
            outlier_share = col.outliers / len(x)
            long_tail = outlier_share > LONG_TAIL_SHARE and skew > SKEW_FOR_LOG and x.min() > -1
            if long_tail:
                suggest("transform", "medium",
                        f"{col.outliers:,} values ({outlier_share:.0%}) beyond 1.5×IQR with skew {skew:.1f}: "
                        f"a long right tail rather than a few outliers. A log scale suits it better than capping.",
                        "log_transform")
            else:
                if outlier_share >= OUTLIER_SHARE_TO_CAP:
                    suggest("outliers", "medium",
                            f"{col.outliers:,} outliers ({outlier_share:.0%}) beyond 1.5×IQR; capping keeps the "
                            f"rows but limits their pull on averages and models.", "cap_outliers")
                if skew > SKEW_FOR_LOG and x.min() > -1:
                    suggest("transform", "low", f"Right-skewed (skew {skew:.1f}); a log scale may suit "
                            f"charts and models.", "log_transform")
        else:
            col.stats["distinct values"] = col.unique
        if 0 < share < MOSTLY_EMPTY:
            method = "median" if abs(skew) > 1 or col.outliers else "mean"
            reason = " (robust to the skew and outliers)" if method == "median" else ""
            suggest("missing", "high" if share >= 0.05 else "medium",
                    f"{missing:,} missing ({share:.1%}); fill with the {method}{reason}.",
                    "fill_missing", numeric=method)
        if charts:
            col.chart = _histogram(x, discrete)
    else:
        counts = values.astype(str).value_counts()
        col.top_values = [(str(k), int(v)) for k, v in counts.head(max_categories).items()]
        col.stats = {"most common": counts.index[0], "most common count": int(counts.iloc[0])}
        if kind == "datetime":
            order = "none"
            if pd.api.types.is_datetime64_any_dtype(series):
                dates = values
            else:
                text = _normalized(values)
                order, examples = _date_order(text)
                dates = _parse_dates(text, dayfirst=order == "day_first")
                if order == "conflict":
                    dates = dates.where(dates.isna())  # can't be read reliably, so no earliest/latest
                    suggest("type", "high", f"Dates mix day-first and month-first order (e.g. {examples[0]!r} "
                            f"and {examples[1]!r}), so some would be read the wrong way round. Fix them at "
                            f"the source, or convert with to_datetime_columns(dayfirst=...) if one order is "
                            f"right for all of them.")
                else:
                    suggest("type", "medium", "Dates stored as text; convert to dates so they sort, filter "
                            "and subtract correctly.", "to_datetime")
            col.stats = {"earliest": dates.min(), "latest": dates.max(), "unparseable": int(dates.isna().sum())}
            if order != "conflict" and any(h in name.lower().replace(" ", "_") for h in DOB_NAME_HINTS):
                suggest("derive", "low", "Looks like a date of birth; derive an age column.", "add_age")
        elif kind == "text":
            col.stats["avg length"] = float(values.astype(str).str.len().mean())
            suggest("note", "low", f"{col.unique:,} distinct values: likely an identifier or free text. "
                    f"Leave it out of encoding and models.")
        elif kind == "categorical" and _is_yes_no(series):
            suggest("type", "medium", "Yes/No answers stored as text; convert to True/False.", "to_boolean")
        elif kind == "categorical" and not numeric_text:  # True/False columns need no encoding
            labels, variants = _standard_spellings(values)
            if variants:
                suggest("type", "medium", f"{variants} label(s) are written more than one way (differing only in "
                        f"case or spaces), so they're counted as different values; standardize them first.",
                        "standardize", labels=tuple(labels))
            categories = len(labels)    # after standardizing, "Male" and "male " are one category
            if categories <= 2:
                suggest("encode", "low", "Two categories; label-encode as 0/1 for modeling.", "encode",
                        method="label")
            elif categories <= 10:
                suggest("encode", "low", f"{categories} categories; one-hot encode for modeling.", "encode",
                        method="onehot")
            else:
                # not label codes: numbering unordered categories invents an order (2 > 1) that linear
                # and distance-based models take literally
                suggest("encode", "low", f"{categories} categories; one-hot encode (adds {categories} columns). "
                        f"Group rare categories into 'Other' first if many are small. Label codes suit "
                        f"tree-based models only: they imply an order the categories don't have.", "encode",
                        method="onehot")

        if 0 < share < MOSTLY_EMPTY and kind == "boolean" and share >= 0.05:
            # a True/False column can't hold a "Missing" label; its gaps can simply stay missing
            suggest("missing", "medium", f"{missing:,} missing ({share:.1%}). A True/False column can keep these "
                    f"as missing values; fill them only if a blank really means one answer.")
        elif 0 < share < MOSTLY_EMPTY and kind in ("categorical", "boolean"):
            method = "most_frequent" if share < 0.05 else "missing"
            why = ("fill with the most common value" if method == "most_frequent"
                   else "fill with a 'Missing' category so the gap stays visible")
            suggest("missing", "high" if share >= 0.05 else "medium", f"{missing:,} missing ({share:.1%}); {why}.",
                    "fill_missing", categorical=method)
        if charts and kind in ("categorical", "boolean"):
            col.chart = _bars(counts.head(max_categories))

    if col.unique == 1 and not (missing_expected and missing):
        # (with meaningful blanks, "value vs blank" is the information, so one value is fine)
        out = [Suggestion(name, "drop", "medium",
                          f"Only one distinct value ({_fmt(values.iloc[0])}); it carries no information.", "drop")]
    elif share >= MOSTLY_EMPTY:
        # a column worth dropping gets no other advice, so the code never transforms a dropped column
        out = [Suggestion(name, "drop", "medium",
                          f"{share:.0%} missing; too sparse to fill reliably. Drop it, or keep only a "
                          f"'has value' flag.", "drop")]
    return col, out


def _already_done(s: Suggestion, names: list[str]) -> bool:
    """Skip steps whose output columns already exist (e.g. after dst.log_transform or encoding)."""
    if s.column is None:
        return False
    if s.action == "log_transform":
        return f"{s.column}_log" in names
    if s.action == "encode":
        return f"{s.column}_label" in names or any(n.startswith(f"{s.column}_") for n in names)
    return False


# ---------- charts ----------

def _png(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["ink"], labelsize=8)


def _histogram(x: pd.Series, discrete: bool = False) -> str:
    fig = Figure(figsize=(3.6, 1.8))
    ax = fig.add_subplot()
    if discrete:
        counts = x.value_counts().sort_index()
        ax.bar([_fmt(v) for v in counts.index], counts.to_numpy(dtype=float), color=COLORS["bar"])
        _style(ax)
        return _png(fig)
    ax.hist(x, bins=min(30, max(5, int(np.sqrt(len(x))))), color=COLORS["bar"], edgecolor="white", linewidth=0.4)
    ax.axvline(float(x.median()), color=COLORS["accent"], linewidth=1.5, linestyle="--", label="median")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    return _png(fig)


def _bars(counts: pd.Series) -> str:
    counts = counts.iloc[::-1]
    labels = [str(k) if len(str(k)) <= 22 else str(k)[:21] + "…" for k in counts.index]
    fig = Figure(figsize=(3.6, 0.35 + 0.22 * len(counts)))
    ax = fig.add_subplot()
    ax.barh(labels, counts.to_numpy(dtype=float), color=COLORS["bar"])
    _style(ax)
    return _png(fig)


def _related_html(pairs: pd.DataFrame | None, skipped: str | None) -> str:
    e = html.escape
    head = ('<h2>Related columns</h2><p class="dst-note">Pairs of columns of any type that are related: '
            "Pearson or Spearman for two numbers, chi-square with Cramér's V for two categories, ANOVA with "
            "eta for one of each. Each pair is tested on the rows where both values are present, and kept when "
            "the adjusted p-value (Benjamini-Hochberg) is below 0.05 and the strength is at least 0.3. "
            "A related pair isn't a cause; a near-perfect one can mean a duplicate column, or a column "
            "recorded after the outcome it predicts.</p>")
    if pairs is None:
        return head + f'<p class="dst-note">Related columns {e(skipped or "not computed")}.</p>'
    if pairs.empty:
        return head + '<p class="dst-note">No pairs are both significant and at least moderately strong.</p>'
    records: list[dict[Hashable, Any]] = pairs.head(RELATED_ROWS_SHOWN).to_dict("records")
    rows = "".join(
        f"<tr><td>{e(str(r['column_1']))}</td><td>{e(str(r['column_2']))}</td><td>{e(str(r['test']))}</td>"
        f"<td>{e(_strength_text(str(r['measure']), float(r['strength'])))}</td>"
        f"<td>{_p_text(float(r['p_adjusted']))}</td><td>{int(r['n']):,}</td><td>{e(str(r['note']))}</td></tr>"
        for r in records)
    more = (f'<p class="dst-note">Showing the strongest {RELATED_ROWS_SHOWN} of {len(pairs):,} pairs; '
            "<code>report.associations</code> has them all.</p>" if len(pairs) > RELATED_ROWS_SHOWN else "")
    return (head + '<div class="dst-scroll"><table class="dst-table"><thead><tr><th>Column</th><th>Column</th>'
            "<th>Test</th><th>Strength</th><th>Adjusted p</th><th>Rows</th><th>Note</th></tr></thead><tbody>"
            + rows + "</tbody></table></div>" + more)


_LEAK = "<b>possible leak</b>"


def _strength_text(measure: str, strength: float) -> str:
    # r and rho have a direction; Cramér's V and eta run from 0 to 1
    return f"{measure} = {strength:+.2f}" if measure in ("r", "rho") else f"{measure} = {strength:.2f}"


def _p_text(p: float) -> str:
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def _heatmap(corr: pd.DataFrame) -> str:
    n = corr.shape[0]
    size = max(3.5, 0.55 * n + 1.5)
    fig = Figure(figsize=(size, size * 0.85))
    ax = fig.add_subplot()
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), [str(c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(n), [str(c) for c in corr.index])
    if n <= 12:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                if np.isnan(v):
                    continue  # undefined: one column doesn't vary where both have values
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else COLORS["ink"])
    fig.colorbar(image, ax=ax, shrink=0.8)
    _style(ax)
    return _png(fig)


# ---------- helpers ----------

def _strong_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    cols = [str(c) for c in corr.columns]
    values = corr.to_numpy(dtype=float)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            r = float(values[i, j])
            if not np.isnan(r) and abs(r) >= threshold:
                pairs.append((a, cols[j], r))
    return sorted(pairs, key=lambda p: -abs(p[2]))


def _fmt(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "–"
        if number.is_integer() and abs(number) < 1e15:
            return f"{int(number):,}"
        if abs(number) >= 1000:
            return f"{number:,.0f}"
        return f"{number:,.3g}" if abs(number) < 1 else f"{number:,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def _one_line(c: ColumnProfile) -> str:
    if c.kind == "numeric":
        return f"{_fmt(c.stats['min'])} to {_fmt(c.stats['max'])}, median {_fmt(c.stats['median'])}"
    if c.kind == "datetime":
        return f"{_fmt(c.stats['earliest'])} to {_fmt(c.stats['latest'])}"
    if c.kind == "empty":
        return "no values"
    if c.top_values:
        top, n = c.top_values[0]
        return f"most common: {top} ({n:,})"
    return ""


def _anchor(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


_CSS = """
.dst-profile { --bg:#f6f7f9; --panel:#fff; --ink:#17202b; --muted:#5d6877; --line:#e2e6eb;
  --accent:#1f6feb; --high:#b42318; --high-bg:#fdecea; --med:#9a5b00; --med-bg:#fff4e0;
  --low:#3b5b8a; --low-bg:#eaf1fb; --code:#f1f3f6;
  color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; max-width:1180px; margin:0 auto; }
@media (prefers-color-scheme: dark) {
  .dst-profile { --bg:#0f141a; --panel:#171e26; --ink:#e6ebf1; --muted:#98a3b1; --line:#29323d;
    --accent:#5c9dff; --high:#ff8a80; --high-bg:#3a1c1a; --med:#f5c77e; --med-bg:#33280f;
    --low:#9cc2ff; --low-bg:#17263a; --code:#1d2530; }
}
.dst-body { margin:0; padding:24px 16px; background:var(--bg, #f6f7f9); }
@media (prefers-color-scheme: dark) { .dst-body { background:#0f141a; } }
.dst-profile h1 { font-size:24px; margin:0 0 14px; }
.dst-profile h2 { font-size:17px; margin:28px 0 10px; }
.dst-profile h3 { font-size:15px; margin:0 0 8px; overflow-wrap:anywhere; }
.dst-tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.dst-tile, .dst-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.dst-label { font-size:12px; color:var(--muted); }
.dst-value { font-size:20px; font-weight:650; font-variant-numeric:tabular-nums; }
.dst-suggestions, .dst-col-notes { list-style:none; padding:0; margin:0; display:grid; gap:6px; }
.dst-suggestions li, .dst-col-notes li { padding:7px 10px; border-radius:8px; }
.dst-col-notes { margin-top:8px; font-size:13px; }
li.dst-high { background:var(--high-bg); color:var(--high); }
li.dst-medium { background:var(--med-bg); color:var(--med); }
li.dst-low { background:var(--low-bg); color:var(--low); }
.dst-badge { display:inline-block; min-width:58px; font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; margin-right:8px; }
.dst-profile code, .dst-code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
.dst-code { background:var(--code); border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto;
  white-space:pre; color:var(--ink); }
.dst-pass { color:var(--low); font-weight:600; }
.dst-fail { color:var(--high); font-weight:700; }
.dst-note, .dst-footer { color:var(--muted); font-size:13px; }
.dst-scroll { overflow-x:auto; }
.dst-table { border-collapse:collapse; width:100%; background:var(--panel); font-variant-numeric:tabular-nums; }
.dst-table th, .dst-table td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
  white-space:nowrap; }
.dst-table th { font-size:12px; color:var(--muted); font-weight:600; }
.dst-table a { color:var(--accent); text-decoration:none; }
.dst-meter { display:inline-block; width:60px; height:6px; background:var(--line); border-radius:3px;
  vertical-align:middle; overflow:hidden; }
.dst-meter span { display:block; height:100%; background:var(--accent); }
.dst-cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:12px; }
.dst-card img, .dst-heatmap { max-width:100%; border-radius:6px; background:#fff; display:block; margin-bottom:8px; }
.dst-heatmap { width:100%; max-width:560px; height:auto; }
.dst-kind { font-weight:400; font-size:12px; color:var(--muted); }
.dst-stats { font-size:12.5px; border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
.dst-stats th { text-align:left; font-weight:500; color:var(--muted); padding:2px 8px 2px 0; }
.dst-stats td { text-align:right; padding:2px 0; }
@media (max-width:600px) { .dst-cards { grid-template-columns:1fr; } }
"""
