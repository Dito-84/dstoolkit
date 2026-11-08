"""Checked by mypy (see [tool.mypy] in pyproject.toml), never run: how a user's code sees dstoolkit's types.

The transformers are imported on first use at run time; type checkers must still see the real classes,
not `object`.
"""

import pandas as pd

import dstoolkit as dst

frame = pd.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
scaled: pd.DataFrame = dst.Scaler(["a"]).fit_transform(frame)
encoder: dst.CategoryEncoder = dst.CategoryEncoder(["c"], min_frequency=0.1)
filler = dst.MissingFiller(numeric="median", add_indicator=["a"])
capper = dst.OutlierCapper(columns=["a"], method="iqr")
dates = dst.DateFeatures(parts=("month",))
report: dst.Profile = dst.profile(frame, charts=False)
