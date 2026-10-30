"""Build the browser app into a folder that any static host (e.g. GitHub Pages) can serve.

    python app/build.py              # -> _site/

The folder contains the page and its script, the Python glue, a freshly built dstoolkit wheel,
the wheels for Excel support and the Titanic sample. Serve it locally with:

    python -m http.server --directory _site 8000
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent

# Excel support, served from the site itself so the page never contacts PyPI (its
# Content-Security-Policy would block that). Pure-Python wheels, pinned; update deliberately.
EXCEL_WHEELS = ["openpyxl==3.1.5", "et_xmlfile==2.0.0"]


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "sample").mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", tmp, str(ROOT)], check=True)
        wheels = list(Path(tmp).glob("dstoolkit-*.whl"))
        if len(wheels) != 1:
            sys.exit(f"Expected one dstoolkit wheel, found {wheels}")
        wheel = shutil.copy(wheels[0], out)

    (out / "wheels").mkdir()
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-q",
                    "-d", str(out / "wheels"), *EXCEL_WHEELS], check=True)
    excel = sorted(f"wheels/{w.name}" for w in (out / "wheels").glob("*.whl"))
    if len(excel) != len(EXCEL_WHEELS):
        sys.exit(f"Expected {len(EXCEL_WHEELS)} Excel wheels, found {excel}")

    script = (APP / "app.js").read_text(encoding="utf-8")
    for placeholder in ("__WHEEL__", "__EXCEL_WHEELS__"):
        if placeholder not in script:
            sys.exit(f"app.js is missing the {placeholder} placeholder")
    script = script.replace("__WHEEL__", Path(wheel).name).replace("__EXCEL_WHEELS__", ",".join(excel))
    (out / "app.js").write_text(script, encoding="utf-8")
    shutil.copy(APP / "index.html", out)
    shutil.copy(APP / "app_core.py", out)
    shutil.copy(ROOT / "data" / "titanic3.csv", out / "sample")
    (out / ".nojekyll").touch()  # serve files as-is on GitHub Pages
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    out = build(parser.parse_args().out)
    print(f"Built the app in {out}:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
