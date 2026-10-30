"""dstoolkit: data cleaning and exploratory analysis helpers for pandas.

Everything is available from the top level:

    import dstoolkit as dst

    df = dst.convert_types(raw)         # yes/no text -> True/False, date text -> dates
    df = dst.clean(df, missing={"numeric": "median"})
    dst.show_distribution_profile(df["age"])
    dst.profile(df).save("report.html")
    dst.validate(df, {"age": {"min": 0, "max": 120}})

or from its module: `dstoolkit.cleaning`, `dstoolkit.exploration`,
`dstoolkit.profiling` and `dstoolkit.validation`.
"""

from . import cleaning, exploration, profiling, validation
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
    to_boolean,
    to_category,
    to_datetime_columns,
    winsorize,
)
from .profiling import ColumnProfile, Profile, Suggestion, profile
from .validation import ColumnRule, Comparison, RuleResult, ValidationError, ValidationReport, validate
from .exploration import (
    add_passed,
    drop_nulls,
    fit_line,
    show_density,
    show_distribution,
    show_distribution_profile,
    show_eda,
    show_eda_table,
    show_counts,
    show_grouped_box,
    show_histogram,
    show_scatter,
)

__version__ = "0.7.0"

__all__ = ["cleaning", "exploration", "profiling", "validation", "__version__",
           *cleaning.__all__, *exploration.__all__, *profiling.__all__, *validation.__all__]
