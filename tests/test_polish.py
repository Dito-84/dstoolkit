"""Small fixes from the November review: clearer errors, empty frames, and the transformers in dir()."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst
from dstoolkit.exploration import _benjamini_hochberg as bh_exploration
from dstoolkit.monitoring import _benjamini_hochberg as bh_monitoring


def test_transformers_are_listed_by_dir_and_match_the_module():
    from dstoolkit import _TRANSFORMERS, transformers
    assert set(_TRANSFORMERS) == set(transformers.__all__)
    assert set(_TRANSFORMERS) <= set(dir(dst))
    assert "profile" in dir(dst)


@pytest.mark.parametrize("bh", [bh_exploration, bh_monitoring])
def test_a_missing_p_value_stays_missing_and_spoils_nothing(bh):
    # by hand for the two real p-values: sorted 0.01, 0.02 -> 0.02, 0.02
    out = bh(np.array([0.01, np.nan, 0.02]))
    assert out[0] == pytest.approx(0.02) and np.isnan(out[1]) and out[2] == pytest.approx(0.02)
    assert np.isnan(bh(np.array([np.nan]))).all()
    assert bh(np.array([0.01, 0.04, 0.03, 0.2])).round(4).tolist() == [0.04, 0.0533, 0.0533, 0.2]


def test_profile_of_a_frame_with_no_rows_says_so_once():
    df = pd.DataFrame({"x": pd.Series([], dtype=float), "c": pd.Series([], dtype=object)})
    suggestions = dst.profile(df, charts=False).suggestions
    assert [(s.column, s.step, s.severity) for s in suggestions] == [(None, "note", "high")]
    assert "no rows" in suggestions[0].message


def test_profile_refuses_names_that_look_the_same():
    df = pd.DataFrame({0: [1.0, 2.0, 3.0], "0": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="look the same"):
        dst.profile(df, charts=False)


def test_profile_charts_without_matplotlib_says_what_to_do():
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import pandas as pd, dstoolkit as dst\n"
        "df = pd.DataFrame({'x': range(30)})\n"
        "assert dst.profile(df, charts=False).columns\n"
        "try:\n"
        "    dst.profile(df)\n"
        "except ImportError as err:\n"
        "    print(err)\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert "pass charts=False" in result.stdout
