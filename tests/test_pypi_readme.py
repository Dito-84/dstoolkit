"""The PyPI README rewrite (.github/scripts/pypi_readme.py)."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "pypi_readme.py"
_spec = importlib.util.spec_from_file_location("pypi_readme", _path)
assert _spec and _spec.loader
pypi_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pypi_readme)

BLOB = "https://github.com/Dito-84/dstoolkit/blob/v1.2.3/"
RAW = "https://raw.githubusercontent.com/Dito-84/dstoolkit/v1.2.3/"


def test_relative_links_point_at_the_release():
    text = "[log](CHANGELOG.md) and [app](./app/) and [nb](notebooks/A.ipynb)"
    assert pypi_readme.absolute(text, "v1.2.3") == (
        f"[log]({BLOB}CHANGELOG.md) and [app]({BLOB}app/) and [nb]({BLOB}notebooks/A.ipynb)")


def test_images_use_the_raw_file_and_nested_links_both_change():
    text = "[![Profile report](docs/profile.png)](docs/titanic_profile.html)"
    assert pypi_readme.absolute(text, "v1.2.3") == (
        f"[![Profile report]({RAW}docs/profile.png)]({BLOB}docs/titanic_profile.html)")


def test_absolute_links_anchors_and_code_are_left_alone():
    text = ("[site](https://dito-84.github.io/dstoolkit/) [section](#exploration) "
            "[mail](mailto:a@example.com) `df[cols](x)`")
    assert pypi_readme.absolute(text, "v1.2.3") == text


def test_the_real_readme_has_no_relative_links_left():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rewritten = pypi_readme.CODE.sub("", pypi_readme.absolute(readme, "v1.2.3"))
    assert pypi_readme.LINK.search(rewritten) is None and pypi_readme.IMAGE.search(rewritten) is None
    assert f"{RAW}docs/profile.png" in rewritten
