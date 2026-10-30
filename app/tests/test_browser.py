"""End-to-end test of the browser app in a real Chromium, loading Python from the Pyodide CDN.

Skipped unless BROWSER_TESTS=1 (it needs internet access and a few minutes).
CI runs it after `python app/build.py`:

    BROWSER_TESTS=1 pytest app/tests/test_browser.py
"""

import functools
import http.server
import os
import threading
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BROWSER_TESTS") != "1",
                                reason="set BROWSER_TESTS=1 to run browser tests")

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(os.environ.get("APP_SITE", ROOT / "_site"))
SHOTS = Path(os.environ.get("APP_SCREENSHOTS", ROOT / "_screenshots"))
LOAD_TIMEOUT = 240_000  # ms; first Pyodide load on a CI machine


@pytest.fixture(scope="module")
def server():
    assert (SITE / "index.html").exists(), "run `python app/build.py` first"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(server, browser):
    context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
    # record anything the Content-Security-Policy blocks
    context.add_init_script("""document.addEventListener("securitypolicyviolation",
        e => (window.__csp = window.__csp || []).push(e.violatedDirective + " " + e.blockedURI));""")
    pg = context.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors
    pg.goto(server)
    pg.wait_for_selector("body[data-ready=true]", timeout=LOAD_TIMEOUT)
    yield pg
    context.close()


def _download(page, button, tmp_path):
    with page.expect_download() as info:
        page.click(button)
    path = tmp_path / info.value.suggested_filename
    info.value.save_as(path)
    return path


def _blocked(page):
    return page.evaluate("window.__csp || []")


def test_pinned_pyodide_matches_its_integrity_hash():
    # if jsDelivr ever served different bytes, the browser would refuse to run them;
    # this says so plainly (and gives the right hash when Pyodide is upgraded)
    import base64
    import hashlib
    import re
    import urllib.request

    page = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    url, pinned = re.search(r'<script src="(https://cdn\.jsdelivr\.net/[^"]+)"\s+integrity="([^"]+)"', page).groups()
    with urllib.request.urlopen(url, timeout=60) as response:
        actual = "sha384-" + base64.b64encode(hashlib.sha384(response.read()).digest()).decode()
    assert pinned == actual, f"{url} hashes to integrity=\"{actual}\""


def test_loads_without_errors(page):
    assert "Ready" in page.inner_text("#status")
    assert page.errors == []
    assert _blocked(page) == []
    page.screenshot(path=SHOTS / "1_ready.png")


def test_sample_is_profiled(page):
    page.click("#sample")
    page.wait_for_selector("body[data-analyzed=true]", timeout=120_000)
    steps = page.locator("#plan input[type=checkbox]")
    assert steps.count() >= 5
    assert "1,309 rows" in page.inner_text("#status")
    notes = page.inner_text("#notes")
    assert "name" in notes and "ticket" in notes
    assert page.locator("#report .dst-profile").count() == 1
    page.screenshot(path=SHOTS / "2_profiled.png", full_page=True)


def test_clean_with_one_step_unticked(page, tmp_path):
    age_fill = page.locator("#plan label", has_text="263 missing").locator("input")
    age_fill.uncheck()
    page.click("#clean")
    page.wait_for_selector("body[data-cleaned=true]", timeout=60_000)
    assert page.is_visible("#result-card")
    page.screenshot(path=SHOTS / "3_cleaned.png", full_page=True)

    csv_path = _download(page, "#dl-csv", tmp_path)
    assert csv_path.name == "titanic3_clean.csv"
    cleaned = pd.read_csv(csv_path)
    assert len(cleaned) == 1309
    assert "cabin" not in cleaned and "boat" not in cleaned           # dropped
    assert cleaned["age"].isna().sum() == 263                           # unticked: left alone
    assert cleaned["fare"].isna().sum() == 0                            # still filled
    assert {"sex_label", "embarked_S", "fare_log"} <= set(cleaned.columns)

    script = _download(page, "#dl-script", tmp_path).read_text()
    assert "pd.read_csv('titanic3.csv'" in script
    assert "columns=['fare']" in script and "'age'" not in script

    report = _download(page, "#dl-report", tmp_path).read_text()
    assert report.startswith("<!doctype html>") and "titanic3.csv" in report
    assert page.errors == []
    assert _blocked(page) == []


def test_excel_upload(page, tmp_path):
    path = tmp_path / "clinic.xlsx"
    pd.DataFrame({"patient": [f"P{i}" for i in range(60)],
                  "age": [30 + i % 40 if i % 7 else None for i in range(60)],
                  "plan": ["Basic", "Plus", "Premium"] * 20}).to_excel(path, index=False)
    page.set_input_files("#file", str(path))
    page.wait_for_function("document.getElementById('status').innerText.includes('clinic.xlsx: 60 rows')",
                           timeout=120_000)
    assert "plan" in page.inner_text("#plan")
    assert page.errors == []
    assert _blocked(page) == []     # openpyxl comes from the pinned Pyodide release, not PyPI


def test_bad_file_shows_friendly_error(page, tmp_path):
    path = tmp_path / "notes.png"
    path.write_bytes(b"not a table")
    page.set_input_files("#file", str(path))
    page.wait_for_selector("#status.error", timeout=30_000)
    assert "Unsupported file type" in page.inner_text("#status")
    assert page.errors == []


def test_phone_layout(server, browser):
    pg = browser.new_page(viewport={"width": 390, "height": 844})
    pg.goto(server)
    width = pg.evaluate("document.documentElement.scrollWidth")
    pg.screenshot(path=SHOTS / "4_phone.png")
    pg.close()
    assert width <= 390
