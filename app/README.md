# Browser app

A drag-and-drop page that profiles and cleans a CSV or Excel file with `dstoolkit`, entirely in the browser. [Pyodide](https://pyodide.org) runs Python (with pandas and matplotlib) in WebAssembly, so the file is never uploaded anywhere.

| File | Purpose |
|---|---|
| `index.html` | The page and its Content-Security-Policy |
| `app.js` | The page's behaviour: loads Pyodide and dstoolkit, handles the dropzone, cleaning plan and downloads |
| `app_core.py` | The Python side: read the file, profile it, apply the chosen steps, write the CSV/script/report |
| `build.py` | Builds `_site/` with the page, a fresh dstoolkit wheel, the pinned Excel wheels and the Titanic sample |
| `tests/test_app_core.py` | Tests the Python side with ordinary Python |
| `tests/test_browser.py` | End-to-end test in a real browser (runs in CI; needs internet) |

## Security

The page promises that your file never leaves it. The browser enforces that, rather than leaving it to trust:

- **Content-Security-Policy** (in `index.html`): the page may load scripts and make requests only to its own site and one pinned Pyodide release on jsDelivr (`/pyodide/v314.0.7/full/`). There are no inline scripts and no `eval`, and it can't submit forms or embed plugins. Even a bug in the page couldn't send data to another server.
- **Subresource Integrity:** `pyodide.js` is loaded with an `integrity` hash, so the browser refuses to run it if jsDelivr ever serves different bytes. Pyodide then checks the packages it loads (pandas, matplotlib) against the hashes in its own lock file.
- **No PyPI at runtime:** Excel support (openpyxl and et_xmlfile) is downloaded at pinned versions when the site is built, and served by the site itself.
- **Tests:** the browser tests fail on any policy violation, and check that the pinned hash still matches the file on jsDelivr. `tests/test_app_core.py` checks, without a network, that the policy, the script tag and `app.js` all name the same Pyodide release.

What's still trusted: jsDelivr serving Pyodide's runtime files (the WebAssembly binary, standard library and lock file), which `pyodide.js` loads and which have no integrity hash of their own. Self-hosting Pyodide would remove that, at the cost of about 25 MB of files in the site.

**Upgrading Pyodide:** change the version in the policy, the script tag and `PYODIDE_INDEX` in `app.js`, then run the browser tests. The hash test fails and prints the new `integrity` value to paste in.

## Run it locally

```bash
python app/build.py
python -m http.server --directory _site 8000
# open http://localhost:8000
```

It needs an internet connection the first time, to download Python from the Pyodide CDN (about 25 MB, then cached by the browser).

## Publishing

`.github/workflows/pages.yml` publishes `_site/` to GitHub Pages on every push to `main`, but skips itself while the repository is private (free accounts can only publish Pages from public repos). After making the repo public: **Settings → Pages → Source: GitHub Actions**, then re-run the **Publish app** workflow.
