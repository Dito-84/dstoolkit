"""Build the browser app into a folder that any static host (e.g. GitHub Pages) can serve.

    python app/build.py              # -> _site/

The folder contains the page and its script, the Python glue, a freshly built dstoolkit wheel
and the Titanic sample. Serve it locally with:

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

    script = (APP / "app.js").read_text(encoding="utf-8")
    if "__WHEEL__" not in script:
        sys.exit("app.js is missing the __WHEEL__ placeholder")
    (out / "app.js").write_text(script.replace("__WHEEL__", Path(wheel).name), encoding="utf-8")
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
