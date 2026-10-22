"""Python side of the dstoolkit browser app.

Runs inside the browser (Pyodide) and is also importable with normal Python,
which is how it is tested. The page calls three functions, passing and
receiving JSON strings:

    analyze(file_name, data_bytes)  -> report HTML + list of suggestions
    clean(selected_ids)             -> cleaned CSV, a Python script, the log
    report_html()                   -> the full report as a standalone page

Nothing here makes a network request: the data stays in the browser.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import PurePath
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

import dstoolkit as dst  # noqa: E402
from dstoolkit.profiling import _CSS, Profile  # noqa: E402

# "?" isn't one of pandas' default missing-value markers but is common in public datasets
EXTRA_NA = ["?"]
MAX_ROWS = 1_000_000

_state: dict[str, Any] = {}


class AppError(ValueError):
    """A problem with the user's file, with a message fit to show them."""


# ---------- reading files ----------

def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AppError("Couldn't read the file's text encoding.")  # pragma: no cover


def _delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Read a CSV/TSV/TXT or Excel (.xlsx) file into a DataFrame."""
    suffix = PurePath(name).suffix.lower()
    if not data:
        raise AppError("The file is empty.")
    if suffix in (".xlsx", ".xlsm"):
        try:
            df = pd.read_excel(io.BytesIO(data), na_values=EXTRA_NA)
        except ImportError as err:  # pragma: no cover - only in a browser without openpyxl
            raise AppError("Excel support didn't load; save the sheet as CSV and try again.") from err
    elif suffix == ".xls":
        raise AppError("Old .xls files aren't supported; save as .xlsx or .csv and try again.")
    elif suffix in (".csv", ".tsv", ".txt", ""):
        text = _decode(data)
        try:
            df = pd.read_csv(io.StringIO(text), sep=_delimiter(text[:65536]), na_values=EXTRA_NA,
                             skipinitialspace=True)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as err:
            raise AppError(f"Couldn't read this as a table: {err}") from err
    else:
        raise AppError(f"Unsupported file type '{suffix}'. Use .csv, .tsv, .txt or .xlsx.")

    if df.shape[1] == 0 or len(df) == 0:
        raise AppError("No rows or columns found in the file.")
    if len(df) > MAX_ROWS:
        raise AppError(f"{len(df):,} rows is more than this page can handle ({MAX_ROWS:,}). "
                       "Use dstoolkit in Python for files this large.")
    if df.columns.duplicated().any():
        df.columns = _dedupe(df.columns.astype(str).tolist())
    return df


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------- the three calls the page makes ----------

def analyze(name: str, data: bytes) -> str:
    """Read and profile a file. Returns JSON for the page."""
    try:
        df = read_table(name, bytes(data))
        report = dst.profile(df, title=PurePath(name).name or "Dataset")
    except AppError as err:
        return json.dumps({"ok": False, "error": str(err)})
    _state.update(name=PurePath(name).name or "data.csv", df=df, report=report)
    suggestions = [{
        "id": i, "column": s.column, "severity": s.severity, "message": s.message,
        "step": s.step, "actionable": s.action is not None,
    } for i, s in enumerate(report.suggestions)]
    return json.dumps({
        "ok": True,
        "rows": report.n_rows, "columns": report.n_columns,
        "css": _CSS, "html": report._fragment(),
        "suggestions": suggestions,
    })


def analyze_file(name: str, path: str) -> str:
    """analyze() for a file the page has written into the browser's in-memory file system."""
    with open(path, "rb") as fh:
        return analyze(name, fh.read())


def clean(selected_json: str) -> str:
    """Apply the chosen suggestions to the loaded data. Returns JSON for the page."""
    if "report" not in _state:
        return json.dumps({"ok": False, "error": "Load a file first."})
    report: Profile = _state["report"]
    ids = set(json.loads(selected_json))
    chosen = [s for i, s in enumerate(report.suggestions) if i in ids and s.action]
    steps = report.code(df_name="df", suggestions=chosen)

    namespace: dict[str, Any] = {"df": _state["df"].copy(), "pd": pd, "dst": dst}
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            exec(steps, namespace)  # code generated by dstoolkit itself from fixed templates
    except Exception as err:  # pragma: no cover - surfaced to the user rather than crashing the page
        return json.dumps({"ok": False, "error": f"Cleaning failed: {err}"})

    cleaned: pd.DataFrame = namespace["df"]
    _state["cleaned"] = cleaned
    return json.dumps({
        "ok": True,
        "rows": len(cleaned), "columns": cleaned.shape[1],
        "csv": cleaned.to_csv(index=False),
        "script": _script(steps),
        "log": log.getvalue().strip() or "No changes selected.",
        "preview": cleaned.head(10).to_html(index=False, border=0, na_rep="", escape=True,
                                            classes="preview-table"),
        "clean_name": _clean_name(_state["name"]),
    })


def report_html() -> str:
    """The full report as a standalone HTML page."""
    report: Profile = _state["report"]
    return report.to_html()


# ---------- helpers ----------

def _clean_name(name: str) -> str:
    stem = PurePath(name).stem or "data"
    return f"{stem}_clean.csv"


def _script(steps: str) -> str:
    """A complete script: read the original file, apply the steps, save the result."""
    name = _state["name"]
    reader = (f"pd.read_excel({name!r}, na_values={EXTRA_NA!r})"
              if PurePath(name).suffix.lower() in (".xlsx", ".xlsm")
              else f"pd.read_csv({name!r}, na_values={EXTRA_NA!r})")
    body = [line for line in steps.splitlines() if not line.startswith("import ")]
    while body and not body[0].strip():
        body.pop(0)
    return "\n".join([
        "# Cleaning steps chosen in the dstoolkit browser app",
        "# pip install git+https://github.com/Dito-84/Data_Science_Toolkit",
        "import pandas as pd",
        "import dstoolkit as dst",
        "",
        f"df = {reader}",
        *body,
        f"df.to_csv({_clean_name(name)!r}, index=False)",
        "",
    ])
