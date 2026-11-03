"""profile(target=...), report.pipeline() and associations(target=...)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import dstoolkit as dst
from dstoolkit.profiling import TargetProfile


def patients(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Made-up records: `readmitted` depends on age and plan; `followup_code` is filled in only
    for readmitted patients (recorded after the outcome); `discharge_note` copies the outcome."""
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 12, n).round()
    plan = rng.choice(["Basic", "Plus", "Premium"], n)
    risk = (age - 60) / 12 + np.where(plan == "Basic", 1.0, -0.5) + rng.normal(0, 1, n)
    readmitted = np.where(risk > 0.6, "yes", "no")
    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        "age": age,
        "income": rng.lognormal(10, 1, n),                                 # long tail: log before the split
        "plan": plan,
        "smoker": rng.choice(["Yes", "No"], n),                            # yes/no text
        "cost_text": [f"{c:,.2f}" for c in rng.uniform(100, 9000, n)],     # numbers stored as text
        "followup_code": np.where(readmitted == "yes", "F1", None),        # blank exactly when not readmitted
        "discharge_note": np.where(readmitted == "yes", "readmit", "home"),  # the answer itself
        "admitted": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "readmitted": readmitted,
    })
    df.loc[:39, "age"] = np.nan
    df.loc[:9, "plan"] = None
    return df


def run(code: str, df: pd.DataFrame) -> dict:
    scope = {"df": df.copy()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code, scope)
    return scope


def test_target_section_finds_leaks_by_value_and_by_blanks():
    df = patients()
    before = df.copy()
    r = dst.profile(df, charts=False, target="readmitted")
    assert isinstance(r.target, TargetProfile)
    assert r.target.task == "classification"
    assert set(r.target.leaks) == {"discharge_note", "followup_code"}
    assert r.target.features["column_1"].eq("readmitted").all()
    assert "followup_code" in set(r.target.missingness["column_2"])
    leak_notes = [s for s in r.suggestions if s.message.startswith("Possible leak")]
    assert {s.column for s in leak_notes} == {"discharge_note", "followup_code"}
    assert all(s.action is None for s in leak_notes)            # never dropped by the cleaning code
    html = r.to_html()
    assert "<h2>Target: readmitted</h2>" in html and "possible leak" in html
    assert "Leak-free modelling starter" in html
    pd.testing.assert_frame_equal(df, before)


def test_target_gets_no_fill_encode_or_outlier_advice():
    df = patients()
    df.loc[:4, "readmitted"] = None
    r = dst.profile(df, charts=False, target="readmitted")
    target_steps = [(s.step, s.action) for s in r.suggestions if s.column == "readmitted"]
    assert ("drop", "drop_missing_target") in target_steps
    assert not any(a in ("fill_missing", "encode", "cap_outliers", "log_transform") for _, a in target_steps)
    assert "df = df.dropna(subset=['readmitted'])" in r.code()
    assert r.target.missing == 5


def test_imbalanced_classes_are_flagged():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=500), "fraud": [1] * 20 + [0] * 480})
    r = dst.profile(df, charts=False, target="fraud")
    assert r.target.task == "classification"
    assert list(r.target.classes.index) == [0, 1]
    note = next(s for s in r.suggestions if s.column == "fraud")
    assert "Imbalanced" in note.message and "4.0%" in note.message


@pytest.mark.parametrize(("values", "expected"), [
    ([0, 1] * 50, "classification"),
    (list(range(5)) * 20, "classification"),
    (list(range(100)), "regression"),                 # many whole numbers: a quantity
    ([0.5, 1.5] * 50, "regression"),                  # not whole numbers
    (["a", "b"] * 50, "classification"),
    ([True, False] * 50, "classification"),
])
def test_task_is_inferred(values, expected):
    df = pd.DataFrame({"x": np.arange(100.0), "y": values})
    assert dst.profile(df, charts=False, related=False, target="y").target.task == expected


def test_task_can_be_set_and_is_checked():
    df = pd.DataFrame({"x": np.arange(100.0), "y": [1, 2, 3, 4] * 25})
    assert dst.profile(df, charts=False, target="y", task="regression").target.task == "regression"
    with pytest.raises(ValueError, match="task must be"):
        dst.profile(df, charts=False, target="y", task="ranking")
    with pytest.raises(KeyError, match="not a column"):
        dst.profile(df, charts=False, target="z")


def test_constant_target_has_nothing_to_predict():
    r = dst.profile(pd.DataFrame({"x": range(20), "y": ["a"] * 20}), charts=False, target="y")
    assert any("nothing to predict" in s.message for s in r.suggestions)
    assert r.target.features.empty


def test_pipeline_needs_a_target():
    with pytest.raises(ValueError, match="target="):
        dst.profile(patients(), charts=False).pipeline()


def test_pipeline_runs_and_keeps_learned_steps_inside_the_pipeline():
    df = patients()
    code = dst.profile(df, charts=False, target="readmitted").pipeline()
    before_split, after_split = code.split("X = df[", 1)
    for learned in ("MissingFiller", "CategoryEncoder", "OutlierCapper"):
        assert learned not in before_split                    # nothing learned on all the rows
    assert "dst.log_transform(df, ['income'])" in before_split
    assert "dst.to_boolean(df, ['smoker'])" in before_split
    assert "pd.to_numeric" not in code or "import pandas" in code
    assert "followup_code" not in after_split.split("\n")[0] and "discharge_note" not in after_split.split("\n")[0]
    assert "# Left out (possible leaks" in code
    assert "patient_id" in code.split("identifiers or free text):")[1].split("\n")[0]
    assert "StratifiedKFold" in code and "scoring='roc_auc'" in code
    scope = run(code, df)
    assert 0.5 < scope["scores"].mean() <= 1.0
    assert scope["baseline"].mean() == pytest.approx(0.5)              # ROC AUC of a constant guess
    assert "from sklearn.dummy import DummyClassifier" in code
    assert list(scope["X"].columns) == ["age", "income", "plan", "smoker", "cost_text"]


def test_pipeline_without_the_leaks_scores_far_below_with_them():
    df = patients()
    code = dst.profile(df, charts=False, target="readmitted").pipeline()
    honest = run(code, df)["scores"].mean()
    leaky = run(code.replace("X = df[['age'", "X = df[['discharge_note', 'age'").replace(
        "dst.CategoryEncoder({", "dst.CategoryEncoder({'discharge_note': 'label', "), df)["scores"].mean()
    assert leaky > 0.99 and honest < 0.95


def test_pipeline_for_regression_and_multiclass():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    group = rng.choice(list("abcdefghijkl"), n)                  # 12 categories: one-hot, not label codes
    df = pd.DataFrame({"x": x, "group": group, "flag": rng.choice([True, False, None], n),
                       "price": 3 * x + 2 * (group == "a") + rng.normal(0, 2, n),   # related, not a copy
                       "tier": rng.choice(["low", "mid", "high"], n)})
    reg = dst.profile(df.drop(columns="tier"), charts=False, target="price").pipeline()
    assert "KFold(5" in reg and "HistGradientBoostingRegressor" in reg
    assert "'group': 'onehot'" in reg
    assert "possible leaks" not in reg
    scores = run(reg, df.drop(columns="tier"))["scores"]                # negative RMSE
    assert -3.0 < scores.mean() < -1.5                                  # the noise in price is 2
    assert run(reg, df.drop(columns="tier"))["baseline"].mean() < scores.mean()   # the average is worse
    assert "DummyRegressor()" in reg
    multi = dst.profile(df.drop(columns="price"), charts=False, target="tier").pipeline()
    assert "scoring='balanced_accuracy'" in multi
    assert "astype('boolean').astype(float)" in multi
    run(multi, df.drop(columns="price"))


def test_associations_with_a_target_tests_only_its_pairs():
    df = patients()
    out = dst.associations(df, target="readmitted", significant_only=False, verbose=False)
    assert out["column_1"].eq("readmitted").all()
    assert "readmitted" not in set(out["column_2"])
    assert len(out) == out["column_2"].nunique()
    restricted = dst.associations(df, columns=["age", "plan"], target="readmitted", significant_only=False,
                                  verbose=False)
    assert set(restricted["column_2"]) <= {"age", "plan"}
    with pytest.raises(ValueError, match="target can't be tested"):
        dst.associations(df, target="patient_id", verbose=False)


def test_many_categories_get_one_hot_not_label_codes():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"city": rng.choice([f"c{i}" for i in range(15)], 300)})
    s = next(s for s in dst.profile(df, charts=False).suggestions if s.action == "encode")
    assert s.options["method"] == "onehot"
    assert "imply an order" in s.message


def test_pipeline_runs_on_messy_data():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(45, 12, n).round(),
        "visits": rng.integers(0, 4, n),
        "zip": [f"{z:05d}" for z in rng.integers(1000, 99999, n)],
        "notes": [None] * 150 + ["follow up"] * 50,                       # mostly empty
        "country": ["US"] * n,                                            # constant
        "date_of_birth": pd.date_range("1950-01-01", periods=n, freq="70D").strftime("%Y-%m-%d"),
        "blank": [np.nan] * n,
        "sex": rng.choice(["F", "M", "f ", None], n),                      # spellings to standardize, blanks
        "churned": rng.choice([0, 1], n),
    })
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)              # duplicates: removed before the split
    code = dst.profile(df, charts=False, target="churned").pipeline()
    assert code.index("dst.remove_duplicates") < code.index("X = df[")
    assert "dst.standardize_labels(df, 'sex'" in code
    scope = run(code, df)
    assert set(scope["X"].columns) == {"age", "visits", "sex"}
    assert len(scope["X"]) == 200


def test_after_outcome_marks_leaks_the_statistics_miss():
    df = patients()
    df["discharge_unit"] = np.random.default_rng(5).choice(["A", "B", "C"], len(df))   # unrelated, but recorded after
    r = dst.profile(df, charts=False, target="readmitted", after_outcome=["discharge_unit"])
    assert r.target.after_outcome == ["discharge_unit"] and "discharge_unit" in r.target.leaks
    note = next(s for s in r.suggestions if s.column == "discharge_unit" and s.step == "drop")
    assert "after the outcome" in note.message and note.action is None
    code = r.pipeline()
    assert "# Left out (recorded after the outcome (after_outcome)): discharge_unit" in code
    assert "discharge_unit" not in code.split("X = df[")[1].split("\n")[0]
    with pytest.raises(ValueError, match="needs target"):
        dst.profile(df, charts=False, after_outcome=["discharge_unit"])
    with pytest.raises(KeyError, match="after_outcome"):
        dst.profile(df, charts=False, target="readmitted", after_outcome=["readmitted"])


def test_missing_flags_are_opt_in():
    df = patients()
    r = dst.profile(df, charts=False, target="readmitted")
    default = r.pipeline()
    assert "_missing'] = " not in default                         # never applied unless asked
    code = r.pipeline(missing_flags=["age"])
    assert "df['age_missing'] = df['age'].isna().astype(int)" in code.split("X = df[")[0]
    scope = run(code, df)
    assert "age_missing" in scope["X"].columns
    with pytest.raises(ValueError, match="possible leaks"):
        r.pipeline(missing_flags=["followup_code"])
    with pytest.raises(ValueError, match="no blanks"):
        r.pipeline(missing_flags=["income"])
    with pytest.raises(TypeError, match="not one string"):
        r.pipeline(missing_flags="age")


def test_fitted_pipeline_survives_saving_and_loading(tmp_path):
    joblib = pytest.importorskip("joblib")
    df = patients()
    scope = run(dst.profile(df, charts=False, target="readmitted").pipeline(missing_flags=["age"]), df)
    X = scope["X"]
    model = scope["model"].fit(X, scope["y"])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    again = joblib.load(path)
    np.testing.assert_array_equal(again.predict_proba(X), model.predict_proba(X))
    assert list(again[:-1].get_feature_names_out()) == list(model[:-1].get_feature_names_out())


def admissions(patients_n: int = 150, per_patient: int = 4, seed: int = 6) -> pd.DataFrame:
    """Several admissions per patient. Readmission depends on a hidden per-patient trait only, and each patient
    has a fixed but meaningless `baseline_score`: a model can memorise patients by it, nothing more."""
    rng = np.random.default_rng(seed)
    trait = rng.random(patients_n) < 0.4
    score = rng.normal(size=patients_n)
    rows = np.repeat(np.arange(patients_n), per_patient)
    return pd.DataFrame({
        "mrn": [f"M{p:05d}" for p in rows],
        "baseline_score": score[rows],
        "age": rng.normal(60, 10, len(rows)).round(),
        "readmitted": np.where(trait[rows] ^ (rng.random(len(rows)) < 0.1), "yes", "no"),
    })


def test_repeated_identifiers_are_flagged():
    df = admissions()
    r = dst.profile(df, charts=False, target="readmitted")
    note = next(s for s in r.suggestions if s.column == "mrn" and "more than one row" in s.message)
    assert "150 value(s)" in note.message and "group='mrn'" in note.message
    unique = dst.profile(patients(), charts=False)                   # one row per patient: no warning
    assert not any("more than one row" in s.message for s in unique.suggestions)
    few = pd.DataFrame({"plan_code": ["A", "B"] * 50, "x": range(100)})   # a category, not an identifier
    assert not any("more than one row" in s.message for s in dst.profile(few, charts=False).suggestions)


def test_grouped_folds_stop_patients_leaking_between_folds():
    df = admissions()
    plain = run(dst.profile(df, charts=False, target="readmitted").pipeline(), df)["scores"].mean()
    report = dst.profile(df, charts=False, target="readmitted", group="mrn")
    assert report.group == "mrn"
    assert not any("more than one row" in s.message for s in report.suggestions)
    code = report.pipeline()
    assert "StratifiedGroupKFold" in code and "groups=groups" in code
    assert "# Left out (the group column: used to split, not to predict): mrn" in code
    grouped = run(code, df)["scores"].mean()
    assert plain > 0.7          # memorised patients: looks good
    assert grouped < 0.6        # honest: the features carry no real signal
    assert plain - grouped > 0.15


def test_grouped_regression_and_bad_group():
    df = admissions().assign(cost=lambda d: d["age"] * 10 + np.random.default_rng(7).normal(0, 50, len(d)))
    code = dst.profile(df.drop(columns="readmitted"), charts=False, target="cost", group="mrn").pipeline()
    assert "GroupKFold(5)" in code
    run(code, df.drop(columns="readmitted"))
    with pytest.raises(KeyError, match="group"):
        dst.profile(df, charts=False, target="cost", group="cost")


def test_pipeline_groups_rare_categories_of_many_valued_columns():
    rng = np.random.default_rng(8)
    n = 600
    common = [f"D{i}" for i in range(8)]
    rare = [f"R{i:03d}" for i in range(120)]                           # many codes seen a handful of times
    dx = np.where(rng.random(n) < 0.7, rng.choice(common, n), rng.choice(rare, n))
    df = pd.DataFrame({"dx": dx, "age": rng.normal(60, 10, n).round(),
                       "readmitted": rng.choice(["yes", "no"], n)})
    code = dst.profile(df, charts=False, target="readmitted").pipeline()
    assert "dst.CategoryEncoder({'dx': 'onehot'}, min_frequency=0.01, drop_original=True)" in code
    scope = run(code, df)
    width = scope["model"].fit(scope["X"], scope["y"])[:-1].transform(scope["X"]).shape[1]
    assert width < 20                                                  # not 128 one-hot columns
