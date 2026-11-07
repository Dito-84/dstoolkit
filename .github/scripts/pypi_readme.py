"""Rewrite README.md's relative links for PyPI, which can't follow them.

Run by the publish workflow just before building, on the release's checkout only;
the README in the repository keeps its relative links. Images point at the raw
file and other links at the file on GitHub, both at the release's tag, so the
PyPI page shows the README as it was for that version.

    python .github/scripts/pypi_readme.py v0.15.1
"""

import re
import sys
from pathlib import Path

REPO = "Dito-84/dstoolkit"
RELATIVE = r"(?![a-z][a-z0-9+.-]*:|#|/)([^)\s]+)"     # no scheme, anchor or leading /
IMAGE = re.compile(r"!\[([^\]]*)\]\(" + RELATIVE + r"\)")
# Link text may itself hold an image: [![alt](src)](target)
LINK = re.compile(r"(?<!!)\[((?:!\[[^\]]*\]\([^)]*\)|[^\[\]])*)\]\(" + RELATIVE + r"\)")
CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)        # code is left exactly as written


def _rewrite(text: str, ref: str) -> str:
    text = IMAGE.sub(lambda m: f"![{m[1]}](https://raw.githubusercontent.com/{REPO}/{ref}/"
                               f"{m[2].removeprefix('./')})", text)
    return LINK.sub(lambda m: f"[{m[1]}](https://github.com/{REPO}/blob/{ref}/{m[2].removeprefix('./')})", text)


def absolute(markdown: str, ref: str) -> str:
    """Code is swapped for placeholders first, so links whose text is code ([`app/`](app/)) still change."""
    code: list[str] = []

    def hide(match: re.Match[str]) -> str:
        code.append(match[0])
        return f"\x00{len(code) - 1}\x00"

    text = _rewrite(CODE.sub(hide, markdown), ref)
    return re.sub(r"\x00(\d+)\x00", lambda m: code[int(m[1])], text)


def main(argv: list[str]) -> None:
    ref = argv[1] if len(argv) > 1 else "main"
    path = Path(__file__).resolve().parents[2] / "README.md"
    path.write_text(absolute(path.read_text(encoding="utf-8"), ref), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
