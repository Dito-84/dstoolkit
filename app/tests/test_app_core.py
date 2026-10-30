"""Tests for the browser app's Python side, run with ordinary Python."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parent.parent
ROOT = APP.parent
sys.path.insert(0, str(APP))

import app_core  # noqa: E402

TITANIC = (ROOT / "data" / "titanic3.csv").read_bytes()


@pytest.fixture(autouse=True)
def fresh_state():
    app_core._state.clear()
    yield
    app_core._state.clear()


def test_analyze_titanic():
    result = json.loads(app_core.analyze("titanic3.csv", TITANIC))
    assert result["ok"]
    assert (result["rows"], result["columns"]) == (1309, 14)
    assert "dst-profile" in result["html"] and ".dst-profile" in result["css"]
    actionable = [s for s in result["suggestions"] if s["actionable"]]
    notes = [s for s in result["suggestions"] if not s["actionable"]]
    assert {s["column"] for s in notes} == {"name", "ticket", "home.dest"}
    assert any(s["column"] == "age" and s["step"] == "missing" for s in actionable)


def test_clean_applies_only_selected_steps():
    analysis = json.loads(app_core.analyze("titanic3.csv", TITANIC))
    drop_ids = [s["id"] for s in analysis["suggestions"] if s["step"] == "drop"]
    result = json.loads(app_core.clean(json.dumps(drop_ids)))
    assert result["ok"]
    cleaned = pd.read_csv(pd.io.common.StringIO(result["csv"]))
    assert "cabin" not in cleaned and "boat" not in cleaned
    assert cleaned["age"].isna().sum() == 263          # filling wasn't selected
    assert result["clean_name"] == "titanic3_clean.csv"


def test_clean_all_and_script_reproduces_result(tmp_path):
    analysis = json.loads(app_core.analyze("titanic3.csv", TITANIC))
    ids = [s["id"] for s in analysis["suggestions"] if s["actionable"]]
    result = json.loads(app_core.clean(json.dumps(ids)))
    cleaned = pd.read_csv(pd.io.common.StringIO(result["csv"]))
    assert cleaned["age"].isna().sum() == 0
    assert {"sex_label", "embarked_S", "fare_log"} <= set(cleaned.columns)

    # The downloadable script, run on the original file, must produce the same data
    (tmp_path / "titanic3.csv").write_bytes(TITANIC)
    (tmp_path / "script.py").write_text(result["script"])
    subprocess.run([sys.executable, "script.py"], cwd=tmp_path, check=True, capture_output=True)
    from_script = pd.read_csv(tmp_path / "titanic3_clean.csv")
    pd.testing.assert_frame_equal(from_script, cleaned)


def test_nothing_selected():
    app_core.analyze("titanic3.csv", TITANIC)
    result = json.loads(app_core.clean("[]"))
    assert result["ok"] and result["log"] == "No changes selected."
    assert result["columns"] == 14


def test_semicolon_and_tab_files():
    for sep, name in ((";", "data.csv"), ("\t", "data.tsv")):
        data = sep.join(["a", "b"]).encode() + b"\n" + sep.join(["1", "x"]).encode() + b"\n"
        result = json.loads(app_core.analyze(name, data))
        assert result["ok"] and result["columns"] == 2, name


def test_latin1_and_duplicate_headers():
    data = "name,name,city\nJosé,1,Bogotá\n".encode("cp1252")
    result = json.loads(app_core.analyze("people.csv", data))
    assert result["ok"]
    assert list(app_core._state["df"].columns) == ["name", "name.1", "city"]  # pandas renames repeats
    assert app_core._state["df"].loc[0, "city"] == "Bogotá"


def test_excel(tmp_path):
    path = tmp_path / "sheet.xlsx"
    pd.DataFrame({"x": [1, 2, None], "g": ["a", "b", "a"]}).to_excel(path, index=False)
    result = json.loads(app_core.analyze("sheet.xlsx", path.read_bytes()))
    assert result["ok"] and result["rows"] == 3
    clean = json.loads(app_core.clean(json.dumps([s["id"] for s in result["suggestions"] if s["actionable"]])))
    assert "pd.read_excel('sheet.xlsx'" in clean["script"]


@pytest.mark.parametrize("name,data,message", [
    ("empty.csv", b"", "empty"),
    ("old.xls", b"x", ".xls"),
    ("photo.png", b"x", "Unsupported"),
    ("headers.csv", b"a,b\n", "No rows"),
])
def test_friendly_errors(name, data, message):
    result = json.loads(app_core.analyze(name, data))
    assert not result["ok"] and message in result["error"]


def test_report_html_is_standalone():
    app_core.analyze("titanic3.csv", TITANIC)
    page = app_core.report_html()
    assert page.startswith("<!doctype html>") and "titanic3.csv" in page


def test_core_makes_no_network_calls():
    source = (APP / "app_core.py").read_text()
    for forbidden in ("urllib", "requests", "http.client", "socket", "pyodide.http", "fetch("):
        assert forbidden not in source


def test_analyze_file_and_preview(tmp_path):
    path = tmp_path / "upload"
    path.write_bytes(TITANIC)
    assert json.loads(app_core.analyze_file("titanic3.csv", str(path)))["ok"]
    preview = json.loads(app_core.clean("[]"))["preview"]
    assert preview.count("<tr") == 11           # header + 10 rows
    assert "<script" not in preview


def test_preview_escapes_cell_values():
    data = b"name,v\n<script>alert(1)</script>,1\n"
    app_core.analyze("x.csv", data)
    preview = json.loads(app_core.clean("[]"))["preview"]
    assert "<script>" not in preview and "&lt;script&gt;" in preview


# ---------- page security (no network needed) ----------

PAGE = (APP / "index.html").read_text(encoding="utf-8")
SCRIPT = (APP / "app.js").read_text(encoding="utf-8")


def _csp() -> dict[str, list[str]]:
    import re
    content = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', PAGE).group(1)
    return {d.split()[0]: d.split()[1:] for d in (part.strip() for part in content.split(";")) if d}


def test_page_has_no_inline_scripts():
    # the policy has no 'unsafe-inline' for scripts, so an inline <script> would be blocked
    import re
    tags = re.findall(r"<script\b[^>]*>", PAGE)
    assert tags and all("src=" in t for t in tags)
    assert "'unsafe-inline'" not in _csp()["script-src"]
    assert "'unsafe-eval'" not in _csp()["script-src"]


def test_policy_allows_only_this_site_and_the_pinned_pyodide_release():
    import re
    csp = _csp()
    assert csp["default-src"] == ["'none'"]
    pinned = re.search(r'<script src="(https://cdn\.jsdelivr\.net/pyodide/[^"]+/full/)pyodide\.js"', PAGE).group(1)
    for directive in ("script-src", "connect-src"):
        hosts = [s for s in csp[directive] if s.startswith("http")]
        assert hosts == [pinned], f"{directive} allows {hosts}"
    assert f'const PYODIDE_INDEX = "{pinned}";' in SCRIPT     # all three name the same release


def test_pyodide_script_is_pinned_with_an_integrity_hash():
    import re
    tag = re.search(r'<script src="https://cdn\.jsdelivr\.net/[^>]+>', PAGE).group(0)
    assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]{64}"', tag), tag
    assert 'crossorigin="anonymous"' in tag


def test_build_fills_in_the_wheel_name(tmp_path):
    subprocess.run([sys.executable, str(APP / "build.py"), "--out", str(tmp_path / "site")], check=True,
                   capture_output=True)
    built = (tmp_path / "site" / "app.js").read_text(encoding="utf-8")
    assert "__WHEEL__" not in built and 'const WHEEL = "dstoolkit-' in built
    assert "__EXCEL_WHEELS__" not in built and "wheels/openpyxl-3.1.5-" in built and "wheels/et_xmlfile-2.0.0-" in built
    served = sorted(p.name.split("-")[0] for p in (tmp_path / "site" / "wheels").glob("*.whl"))
    assert served == ["et_xmlfile", "openpyxl"]
    assert (tmp_path / "site" / "index.html").read_text(encoding="utf-8") == PAGE
