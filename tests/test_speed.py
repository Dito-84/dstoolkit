"""Speed-ups that must not change any answer."""

import pandas as pd

import dstoolkit.cleaning as cleaning


def test_label_spellings_compare_each_distinct_value_once(monkeypatch):
    calls = []
    original = cleaning._label_key
    monkeypatch.setattr(cleaning, "_label_key", lambda v: calls.append(v) or original(v))
    values = pd.Series(["Male", "male ", "Female", "MALE", None] * 20_000)
    labels, variants = cleaning._standard_spellings(values)
    assert (labels, variants) == (["Female", "Male"], 1)     # "Male" is the most common spelling of its label
    assert len(calls) == 4                                   # four distinct spellings, not 80,000 values


def test_label_spellings_keep_the_most_common_spelling():
    values = pd.Series(["north"] * 3 + ["North"] * 5 + [" NORTH"] + ["South"] * 2)
    assert cleaning._standard_spellings(values) == (["North", "South"], 1)


def test_numbers_stored_as_text_weighs_each_value_by_its_count():
    from dstoolkit.profiling import _numbers_stored_as_text
    # 96 rows are numbers and 4 aren't: 96% of rows, although only 2 of 6 distinct values are numbers
    series = pd.Series(["1,200"] * 48 + ["3"] * 48 + ["a", "b", "c", "d"], name="amount")
    assert _numbers_stored_as_text(series)
    assert not _numbers_stored_as_text(pd.Series(["1"] * 90 + ["x"] * 10, name="amount"))
