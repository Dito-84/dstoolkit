# Browser app

A drag-and-drop page that profiles and cleans a CSV or Excel file with `dstoolkit`, entirely in the browser. [Pyodide](https://pyodide.org) runs Python (with pandas and matplotlib) in WebAssembly, so the file is never uploaded anywhere.

| File | Purpose |
|---|---|
| `index.html` | The page: loads Pyodide and dstoolkit, handles the dropzone, cleaning plan and downloads |
| `app_core.py` | The Python side: read the file, profile it, apply the chosen steps, write the CSV/script/report |
| `build.py` | Builds `_site/` with the page, a fresh dstoolkit wheel and the Titanic sample |
| `tests/test_app_core.py` | Tests the Python side with ordinary Python |
| `tests/test_browser.py` | End-to-end test in a real browser (runs in CI; needs internet) |

## Run it locally

```bash
python app/build.py
python -m http.server --directory _site 8000
# open http://localhost:8000
```

It needs an internet connection the first time, to download Python from the Pyodide CDN (about 25 MB, then cached by the browser).

## Publishing

`.github/workflows/pages.yml` publishes `_site/` to GitHub Pages on every push to `main`, but skips itself while the repository is private (free accounts can only publish Pages from public repos). After making the repo public: **Settings → Pages → Source: GitHub Actions**, then re-run the **Publish app** workflow.
