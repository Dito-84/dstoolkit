"""clean_column_names() and the profile's checks for messy names and notes rows at the end of a file."""
import io

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst

EXPORT_NAMES = ["Complication Excluded Due to COVID-19 (Yes/No) ", "Number of THAs Performed \n(0, 1, or 2) ",
                "Age", "age ", "!!!", 3]


@pytest.mark.parametrize(("case", "expected"), [
    ("snake", ["complication_excluded_due_to_covid_19_yes_no", "number_of_thas_performed_0_1_or_2",
               "age", "age_2", "column_5", 3]),
    ("pascal", ["ComplicationExcludedDueToCovid19YesNo", "NumberOfThasPerformed01Or2", "Age", "Age_2",
                "column_5", 3]),
    ("strip", ["Complication Excluded Due to COVID-19 (Yes/No)", "Number of THAs Performed (0, 1, or 2)",
               "Age", "age", "!!!", 3]),
])
def test_clean_column_names(case, expected):
    df = pd.DataFrame([range(6)], columns=EXPORT_NAMES)
    out = dst.clean_column_names(df, case, verbose=False)
    assert list(out.columns) == expected
    assert list(df.columns) == EXPORT_NAMES                                # the input is untouched
    assert out.iloc[0].tolist() == list(range(6))


def test_clean_column_names_reports_and_never_duplicates(caplog):
    df = pd.DataFrame(columns=["a b", "a_b", "A-B", "ok"])
    with caplog.at_level("INFO", logger="dstoolkit"):
        out = dst.clean_column_names(df)
    assert list(out.columns) == ["a_b_2", "a_b", "a_b_3", "ok"]          # the clean "a_b" keeps its name
    assert "Renamed 2 column(s)" in caplog.text
    with caplog.at_level("INFO", logger="dstoolkit"):
        dst.clean_column_names(out)
    assert "already clean" in caplog.text
    with pytest.raises(ValueError, match="case must be"):
        dst.clean_column_names(df, "camel")


def export(rows: int = 200, notes: tuple[str, ...] = ("Note: ages are at admission", "Source: made up")) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Patient Age ": rng.integers(20, 90, rows), "Ward\n(code)": rng.choice(["A", "B"], rows),
                       "LOS": rng.integers(1, 9, rows), "Score": rng.normal(size=rows).round(3)})
    return df.to_csv(index=False) + "".join(f"{note},,,\n" for note in notes)


def test_notes_rows_at_the_end_are_found_left_out_and_dropped_first():
    raw = pd.read_csv(io.StringIO(export()))
    assert raw["Patient Age "].dtype != "int64"                         # the notes turned ages into text
    r = dst.profile(raw, charts=False, related=False)
    first = r.suggestions[0]
    assert first.action == "drop_rows" and first.options["rows"] == (200, 201)
    assert "The last 2 row(s) are almost empty" in first.message
    assert r.n_rows == 202
    los = next(c for c in r.columns if c.name == "LOS")
    assert los.missing == 0                                             # notes rows aren't counted as blanks
    code = r.code()
    lines = code.splitlines()
    assert lines[3] == "df = df.drop(index=[200, 201])   # notes at the end of the file"
    assert lines[-1].startswith("df = dst.clean_column_names(df)")    # renamed last, after the old names are used
    scope = {"df": raw.copy()}
    exec(code, scope)
    assert len(scope["df"]) == 200 and "patient_age" in scope["df"].columns
    assert scope["df"]["patient_age"].dtype.kind in "if"


def test_messy_names_get_a_suggestion_and_clean_ones_dont():
    r = dst.profile(pd.read_csv(io.StringIO(export(notes=()))), charts=False, related=False)
    names = [s for s in r.suggestions if s.action == "clean_column_names"]
    assert len(names) == 1 and "2 column name(s) have hidden spaces or line breaks" in names[0].message
    assert not any(s.action == "drop_rows" for s in r.suggestions)
    tidy = dst.profile(dst.clean_column_names(pd.read_csv(io.StringIO(export(notes=()))), verbose=False),
                       charts=False, related=False)
    assert not any(s.action == "clean_column_names" for s in tidy.suggestions)


@pytest.mark.parametrize("case", ["sparse data", "too many", "too small"])
def test_notes_rows_are_not_guessed(case):
    rng = np.random.default_rng(1)
    if case == "sparse data":      # blanks everywhere: the last rows aren't special
        df = pd.DataFrame(rng.normal(size=(200, 5))).mask(rng.random((200, 5)) < 0.85)
    elif case == "too many":       # a third of the file "empty": not a footer
        df = pd.DataFrame(rng.normal(size=(150, 5)))
        df.iloc[100:, 1:] = np.nan
    else:                          # under 20 rows: too few to tell
        df = pd.DataFrame(rng.normal(size=(15, 5)))
        df.iloc[-1, 1:] = np.nan
    assert not any(s.action == "drop_rows" for s in dst.profile(df, charts=False, related=False).suggestions)


def test_pipeline_drops_notes_rows_before_the_split():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "group": rng.choice(["a", "b", "c"], n),
                       "y": rng.choice(["yes", "no"], n)})
    raw = pd.read_csv(io.StringIO(df.to_csv(index=False) + "Totals,,\n"))
    code = dst.profile(raw, charts=False, related=False, target="y").pipeline()
    assert code.index("df = df.drop(index=[300])") < code.index("X = df[")
    scope = {"df": raw.copy()}
    exec(code, scope)
    assert len(scope["X"]) == 300
