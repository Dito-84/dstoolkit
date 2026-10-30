"""The scikit-learn transformers: learn on training data, apply anywhere, work in pipelines."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

import dstoolkit as dst  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def train():
    return pd.DataFrame({"age": [20.0, 30.0, np.nan, 50.0],
                         "fare": [10.0, 12.0, 11.0, 13.0],
                         "city": ["NY", "LA", "NY", None]})


@pytest.fixture
def test():
    # very different numbers, so anything learned from this data would show
    return pd.DataFrame({"age": [np.nan, 90.0], "fare": [500.0, np.nan], "city": [None, "SF"]})


# ---------- no leakage: what's learned comes from the training data only ----------

def test_missing_filler_uses_training_values_on_new_data(train, test):
    filler = dst.MissingFiller(numeric="mean").fit(train)
    assert filler.fill_values_ == {"age": pytest.approx(100 / 3), "fare": 11.5, "city": "NY"}
    out = filler.transform(test)
    assert out["age"].tolist() == [pytest.approx(100 / 3), 90.0]   # not the test mean (90)
    assert out["fare"].tolist() == [500.0, 11.5]
    assert out["city"].tolist() == ["NY", "SF"]
    assert test["age"].isna().sum() == 1                             # input unchanged


def test_outlier_capper_uses_training_bounds(train, test):
    capper = dst.OutlierCapper(columns=["fare"]).fit(train)
    low, high = capper.bounds_["fare"]
    assert capper.transform(test)["fare"].iloc[0] == high            # 500 capped at the training bound
    assert high < 20


def test_scaler_uses_training_range(train, test):
    scaler = dst.Scaler({"fare": "minmax", "age": "zscore"}).fit(train)
    out = scaler.transform(test)
    assert out["fare"].iloc[0] == pytest.approx((500 - 10) / 3)      # beyond 1: outside the training range
    ages = train["age"].dropna()
    assert out["age"].iloc[1] == pytest.approx((90 - ages.mean()) / ages.std(ddof=0))
    assert np.isnan(out["age"].iloc[0])                              # missing stays missing


def test_encoder_keeps_training_categories_and_ignores_unseen(train, test):
    enc = dst.CategoryEncoder({"city": "onehot"}).fit(train)
    out = enc.transform(test)
    assert [c for c in out.columns if c.startswith("city_")] == ["city_LA", "city_NY"]   # no city_SF
    assert out[["city_LA", "city_NY"]].to_numpy().tolist() == [[0, 0], [0, 0]]          # missing and unseen
    labels = dst.CategoryEncoder({"city": "label"}).fit(train).transform(test)
    assert labels["city_label"].isna().all()


def test_encoder_can_refuse_unseen_categories(train, test):
    enc = dst.CategoryEncoder(["city"], handle_unknown="error").fit(train)
    with pytest.raises(ValueError, match=r"weren't in the training data: \['SF'\]"):
        enc.transform(test)


# ---------- same results as the cleaning functions on the data they were fitted on ----------

def test_fit_transform_matches_the_cleaning_functions(train):
    kw = {"verbose": False}
    pd.testing.assert_frame_equal(dst.MissingFiller(numeric="median").fit_transform(train),
                                  dst.fill_missing(train, numeric="median", drop_empty_rows=False, **kw))
    pd.testing.assert_frame_equal(dst.OutlierCapper(columns=["fare"]).fit_transform(train),
                                  dst.handle_outliers(train, ["fare"], action="cap", **kw))
    pd.testing.assert_frame_equal(dst.Scaler({"age": "zscore", "fare": "minmax"}).fit_transform(train),
                                  dst.normalize(train, {"age": "zscore", "fare": "minmax"}, **kw))
    spec = {"city": "onehot"}
    pd.testing.assert_frame_equal(dst.CategoryEncoder(spec).fit_transform(train),
                                  dst.encode_categorical(train, spec, **kw))


# ---------- scikit-learn rules ----------

def test_works_in_a_pipeline_with_cross_validation():
    titanic = pd.read_csv(DATA / "titanic3.csv", na_values="?")
    X, y = titanic[["pclass", "sex", "age", "fare", "embarked"]], titanic["survived"]
    model = make_pipeline(
        dst.MissingFiller(numeric="median", categorical="most_frequent"),
        dst.OutlierCapper(columns=["fare"]),
        dst.CategoryEncoder({"sex": "label", "embarked": "onehot"}, drop_original=True),
        dst.Scaler({"age": "zscore", "fare": "zscore"}),
        LogisticRegression(max_iter=1000),
    )
    # shuffled folds: titanic3.csv is sorted by passenger class, so unshuffled folds are lopsided
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=folds)   # each fold refits the cleaning on its training rows
    assert scores.min() > 0.7

    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
    model.fit(X_train, y_train)
    filler, scaler = model.steps[0][1], model.steps[3][1]
    assert filler.fill_values_["age"] == X_train["age"].median()     # learned from training rows only
    assert scaler.stats_["fare"][0] != pytest.approx(X["fare"].mean())


def test_feature_names_and_column_transformer(train):
    enc = dst.CategoryEncoder({"city": "onehot"}, drop_original=True).fit(train)
    assert list(enc.get_feature_names_out()) == list(enc.transform(train).columns)
    ct = ColumnTransformer([("scale", dst.Scaler(["fare"]), ["fare"])], remainder="passthrough")
    ct.set_output(transform="pandas")
    out = ct.fit_transform(train)
    assert list(out.columns) == ["scale__fare", "remainder__age", "remainder__city"]


def test_clone_pickle_and_params(train):
    filler = dst.MissingFiller(numeric="median", columns=["age"])
    assert filler.get_params() == {"numeric": "median", "categorical": "most_frequent", "columns": ["age"]}
    fitted = filler.fit(train)
    assert not hasattr(clone(fitted), "fill_values_")                # a clone is unfitted
    again = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(again.transform(train), fitted.transform(train))


def test_rows_and_input_are_never_changed(train):
    before = train.copy()
    for t in (dst.MissingFiller(), dst.OutlierCapper(), dst.CategoryEncoder(["city"]), dst.Scaler(["fare"])):
        out = t.fit_transform(train)
        assert list(out.index) == list(train.index)
    pd.testing.assert_frame_equal(train, before)


def test_clear_errors(train):
    with pytest.raises(NotFittedError):
        dst.Scaler(["fare"]).transform(train)
    with pytest.raises(TypeError, match="works on pandas DataFrames"):
        dst.Scaler(["fare"]).fit(train.to_numpy())
    with pytest.raises(KeyError, match="fitted with columns that aren't here"):
        dst.Scaler(["fare"]).fit(train).transform(train.drop(columns="fare"))
    with pytest.raises(TypeError, match="must be numeric"):
        dst.Scaler(["city"]).fit(train)
    with pytest.raises(ValueError, match="numeric must be"):
        dst.MissingFiller(numeric="knn").fit(train)
    with pytest.raises(TypeError, match="not one string"):
        dst.CategoryEncoder("city").fit(train)


def test_dates_and_true_false_gaps_follow_fill_missing(train):
    df = train.assign(when=pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04"]),
                      done=pd.array([True, None, False, True], dtype="boolean"))
    out = dst.MissingFiller(categorical="missing").fit_transform(df)
    assert out["when"].isna().sum() == 1 and out["done"].isna().sum() == 1
    assert out["city"].tolist()[-1] == "Missing"


def test_integer_columns_switch_to_decimals_when_needed():
    df = pd.DataFrame({"n": pd.array([1, 2, None], dtype="Int64")})
    assert str(dst.MissingFiller().fit_transform(df)["n"].dtype) == "Float64"


def test_import_dstoolkit_does_not_need_scikit_learn(monkeypatch):
    # the transformers are imported on first use only
    for name in [m for m in sys.modules if m == "dstoolkit" or m.startswith(("dstoolkit.", "sklearn"))]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    import dstoolkit
    assert dstoolkit.fill_missing is not None
    with pytest.raises(ImportError, match=r"dstoolkit\[sklearn\]"):
        _ = dstoolkit.Scaler
    with pytest.raises(AttributeError):
        _ = dstoolkit.NotAThing
