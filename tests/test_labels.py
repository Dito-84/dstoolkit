"""standardize_labels(): one spelling per label, and nothing changed without being reported."""

import pandas as pd
import pytest

import dstoolkit as dst


@pytest.fixture
def survey():
    return pd.DataFrame({
        "sex": ["Female", "f", " FEMALE ", "woman", "M", "man", "male", None, "unknown", "x"],
        "plan": pd.Categorical(["Basic", "basic", "Plus", "plus  ", "Basic", "Plus", "BASIC", "Plus", "Plus", None]),
    })


SEX = {"Female": ["f", "woman"], "Male": ["m", "man"]}


def test_spellings_become_the_standard_label(survey, capsys):
    before = survey.copy()
    out = dst.standardize_labels(survey, "sex", SEX)
    assert out["sex"].tolist()[:7] == ["Female"] * 4 + ["Male"] * 3
    assert pd.isna(out.loc[7, "sex"])                       # missing stays missing
    pd.testing.assert_frame_equal(survey, before)          # input unchanged
    assert "'sex': 6 value(s) relabeled" in capsys.readouterr().out


def test_unmatched_values_are_kept_and_reported_never_erased_silently(survey, capsys):
    # The 2023 draft this replaces set anything it didn't recognise (e.g. "F", "woman") to blank
    out = dst.standardize_labels(survey, "sex", SEX)
    assert out["sex"].tolist()[8:] == ["unknown", "x"]
    assert "2 matched no label and were left as they were: 'unknown' (1), 'x' (1)" in capsys.readouterr().out


def test_unmatched_can_be_made_missing_or_raise(survey, capsys):
    out = dst.standardize_labels(survey, "sex", SEX, unmatched="missing")
    assert out["sex"].isna().sum() == 3
    assert "were made missing" in capsys.readouterr().out
    with pytest.raises(ValueError, match="match no label: 'unknown'"):
        dst.standardize_labels(survey, "sex", SEX, unmatched="raise")


def test_a_list_fixes_case_and_spaces_and_keeps_category_type(survey):
    out = dst.standardize_labels(survey, "plan", ["Basic", "Plus"], verbose=False)
    assert out["plan"].tolist()[:9] == ["Basic", "Basic", "Plus", "Plus", "Basic", "Plus", "Basic", "Plus", "Plus"]
    assert isinstance(out["plan"].dtype, pd.CategoricalDtype)
    assert sorted(out["plan"].cat.categories) == ["Basic", "Plus"]


def test_several_columns_and_string_dtype():
    df = pd.DataFrame({"a": pd.array(["yes ", "No"], dtype="string"), "b": ["NO", "Yes"]})
    out = dst.standardize_labels(df, ["a", "b"], ["Yes", "No"], verbose=False)
    assert out["a"].tolist() == ["Yes", "No"] and out["b"].tolist() == ["No", "Yes"]
    assert str(out["a"].dtype) == "string"


def test_mistakes_in_the_mapping_raise_before_data_is_read(survey):
    with pytest.raises(ValueError, match="listed for both 'Female' and 'Male'"):
        dst.standardize_labels(survey, "sex", {"Female": ["f", "x"], "Male": [" X"]})
    with pytest.raises(ValueError, match="empty"):
        dst.standardize_labels(survey, "sex", {})
    with pytest.raises(TypeError, match="not one string"):
        dst.standardize_labels(survey, "sex", "Female")
    with pytest.raises(ValueError, match="unmatched"):
        dst.standardize_labels(survey, "sex", SEX, unmatched="drop")
    with pytest.raises(KeyError):
        dst.standardize_labels(survey, "nope", SEX)
    with pytest.raises(TypeError, match="to_boolean"):
        dst.standardize_labels(pd.DataFrame({"n": [1, 0]}), "n", ["1"])


def test_profile_suggests_it_and_the_code_runs():
    plans = ["Basic", "basic", "Plus", "plus  ", "Basic", "Plus", "BASIC", "Plus", "Premium", "Plus"]
    df = pd.DataFrame({"plan": plans * 3, "amount": range(30)})
    report = dst.profile(df, charts=False)
    notes = {s.action: s for s in report.suggestions if s.column == "plan"}
    assert "2 label(s) are written more than one way" in notes["standardize"].message
    assert notes["encode"].message.startswith("3 categories")     # counted after standardizing
    code = report.code()
    assert code.index("dst.standardize_labels(df, 'plan', ['Basic', 'Plus', 'Premium'])") < code.index("encode")
    namespace = {"df": df.copy(), "pd": pd, "dst": dst}
    exec(code, namespace)
    assert set(namespace["df"]["plan"]) == {"Basic", "Plus", "Premium"}


def test_profile_does_not_suggest_it_when_labels_are_consistent():
    df = pd.DataFrame({"plan": ["Basic", "Plus", "Premium"] * 10})
    assert "standardize" not in {s.action for s in dst.profile(df, charts=False).suggestions}
