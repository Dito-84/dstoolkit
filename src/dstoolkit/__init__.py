"""dstoolkit: data cleaning and exploratory analysis helpers for pandas.

Everything is available from the top level:

    import dstoolkit as dst

    df = dst.convert_types(raw)         # yes/no text -> True/False, date text -> dates
    df = dst.clean(df, missing={"numeric": "median"})
    dst.show_distribution_profile(df["age"])
    dst.profile(df).save("report.html")
    dst.validate(df, {"age": {"min": 0, "max": 120}})

or from its module: `dstoolkit.cleaning`, `dstoolkit.exploration`,
`dstoolkit.profiling` and `dstoolkit.validation`. The scikit-learn
transformers (MissingFiller, OutlierCapper, CategoryEncoder, Scaler) are in
`dstoolkit.transformers` and need the optional scikit-learn extra.
"""

from . import cleaning, exploration, profiling, validation
from ._messages import get_message_output, set_message_output
from .cleaning import (
    add_age_column,
    clean,
    convert_types,
    encode_categorical,
    fill_missing,
    find_outliers,
    handle_outliers,
    log_transform,
    normalize,
    remove_duplicates,
    standardize_labels,
    to_boolean,
    to_category,
    to_datetime_columns,
    winsorize,
)
from .exploration import (
    add_passed,
    associations,
    drop_nulls,
    fit_line,
    show_counts,
    show_density,
    show_distribution,
    show_distribution_profile,
    show_eda,
    show_eda_table,
    show_grouped_box,
    show_histogram,
    show_scatter,
)
from .profiling import ColumnProfile, Profile, Suggestion, profile
from .validation import ColumnRule, Comparison, RuleResult, ValidationError, ValidationReport, validate

__version__ = "0.8.0"

# The transformers need scikit-learn (an optional extra), so they're imported on
# first use: `dst.Scaler` works when scikit-learn is installed, and `import
# dstoolkit` never needs it.
_TRANSFORMERS = ("MissingFiller", "OutlierCapper", "CategoryEncoder", "Scaler")


def __getattr__(name: str) -> object:
    if name in _TRANSFORMERS:
        from . import transformers
        return getattr(transformers, name)
    raise AttributeError(f"module 'dstoolkit' has no attribute {name!r}")

__all__ = ["cleaning", "exploration", "profiling", "validation", "__version__",
           "set_message_output", "get_message_output",
           *cleaning.__all__, *exploration.__all__, *profiling.__all__, *validation.__all__]
