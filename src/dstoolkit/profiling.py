"""One-call dataset report: overview, per-column profiles, correlations and
suggested cleaning code.

    import dstoolkit as dst

    report = dst.profile(df, title="Titanic passengers")
    report                      # shows inline in Jupyter
    report.save("report.html")  # one self-contained file to share
    report.suggestions          # the recommended cleaning steps as data
    print(report.code())        # ...or as ready-to-run dstoolkit code

The report never changes the DataFrame it describes.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .cleaning import DOB_NAME_HINTS, _bounds

__all__ = ["profile", "Profile", "ColumnProfile", "Suggestion"]

Kind = Literal["numeric", "categorical", "boolean", "datetime", "text", "empty"]
Severity = Literal["high", "medium", "low"]

DATE_PATTERN = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})([ T]\d{1,2}:\d{2}.*)?\s*$")
MOSTLY_EMPTY = 0.60          # share missing above which a column is suggested for dropping
OUTLIER_SHARE_TO_CAP = 0.01  # share of outliers above which capping is suggested
LONG_TAIL_SHARE = 0.10       # above this share, "outliers" are a long tail: suggest log, not capping
SKEW_FOR_LOG = 1.0           # skewness above which a log transform is suggested
DISCRETE_MAX_UNIQUE = 10     # numeric columns with few values (counts, codes) skip outlier advice
TEXT_MIN_UNIQUE = 50         # text columns with more distinct values than this...
TEXT_MIN_RATIO = 0.2         # ...and this share of distinct values are treated as free text / IDs
HIGH_CORRELATION = 0.9

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


class Profile:
    """A dataset report. Display it in Jupyter, save it as HTML, or read its parts."""

    def __init__(self, title: str, n_rows: int, n_columns: int, duplicates: int, missing_cells: int,
                 memory_bytes: int, columns: list[ColumnProfile], correlations: pd.DataFrame | None,
                 correlation_chart: str | None, suggestions: list[Suggestion]) -> None:
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
        d = df_name
        chosen = self.suggestions if suggestions is None else list(suggestions)
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
            elif action == "add_age":
                lines += [f"{d} = dst.add_age_column({d}, dob_column={i.column!r})" for i in items]

        # one encode call is easier to read than one per method
        encodes = [line for line in lines if "dst.encode_categorical(" in line]
        if len(encodes) > 1:
            merged = ", ".join(line.split("{", 1)[1].rsplit("}", 1)[0] for line in encodes)
            first = lines.index(encodes[0])
            lines = [line for line in lines if line not in encodes]
            lines.insert(first, f"{d} = dst.encode_categorical({d}, {{{merged}}})")

        header = ["import pandas as pd"] if any("pd." in line for line in lines) else []
        header += ["import dstoolkit as dst", ""]
        return "\n".join(header + (lines or ["# No cleaning steps suggested"]))

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
        parts = [f'<div class="dst-profile"><h1>{e(self.title)}</h1>',
                 '<div class="dst-tiles">' + "".join(
                     f'<div class="dst-tile"><div class="dst-label">{e(k)}</div><div class="dst-value">{e(v)}</div></div>'
                     for k, v in tiles) + "</div>"]

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
                             '<table class="dst-table"><thead><tr><th>Column</th><th>Column</th><th>r</th></tr></thead><tbody>'
                             + "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{r:+.2f}</td></tr>" for a, b, r in pairs)
                             + "</tbody></table>")
        parts.append('<p class="dst-footer">Generated by dstoolkit.profile()</p></div>')
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
            max_categories: int = 10, missing_ok: Iterable[str] = ()) -> Profile:
    """Profile a DataFrame: overview, per-column statistics, correlations and
    suggested cleaning steps.

    charts: draw a small chart for each column and a correlation heatmap
    max_categories: how many of the most common values to show per text column
    missing_ok: columns where a blank means something (e.g. "no complication",
                "not readmitted"), so they get no suggestions to fill or drop them
    """
    if df.shape[1] == 0:
        raise ValueError("DataFrame has no columns")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Column name(s) {dupes} appear more than once; rename duplicates first")

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

    order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: (STEP_ORDER.get(s.step, 9), order[s.severity]))

    return Profile(title=title, n_rows=len(df), n_columns=df.shape[1], duplicates=duplicates,
                   missing_cells=int(df.isna().sum().sum()),
                   memory_bytes=int(df.memory_usage(deep=True).sum()), columns=columns,
                   correlations=correlations, correlation_chart=corr_chart, suggestions=suggestions)


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
            dates = values if pd.api.types.is_datetime64_any_dtype(series) else \
                pd.to_datetime(values, errors="coerce", format="mixed")
            col.stats = {"earliest": dates.min(), "latest": dates.max(), "unparseable": int(dates.isna().sum())}
            if any(h in name.lower().replace(" ", "_") for h in DOB_NAME_HINTS):
                suggest("derive", "low", "Looks like a date of birth; derive an age column.", "add_age")
        elif kind == "text":
            col.stats["avg length"] = float(values.astype(str).str.len().mean())
            suggest("note", "low", f"{col.unique:,} distinct values: likely an identifier or free text. "
                    f"Leave it out of encoding and models.")
        elif kind == "categorical" and not numeric_text:  # True/False columns need no encoding
            if col.unique <= 2:
                suggest("encode", "low", "Two categories; label-encode as 0/1 for modeling.", "encode",
                        method="label")
            elif col.unique <= 10:
                suggest("encode", "low", f"{col.unique} categories; one-hot encode for modeling.", "encode",
                        method="onehot")
            else:
                suggest("encode", "low", f"{col.unique} categories; label-encode, or group rare ones "
                        f"before one-hot encoding.", "encode", method="label")

        if 0 < share < MOSTLY_EMPTY and kind in ("categorical", "boolean"):
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
.dst-note, .dst-footer { color:var(--muted); font-size:13px; }
.dst-scroll { overflow-x:auto; }
.dst-table { border-collapse:collapse; width:100%; background:var(--panel); font-variant-numeric:tabular-nums; }
.dst-table th, .dst-table td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
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
