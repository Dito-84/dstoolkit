"""dstoolkit: data cleaning and exploratory analysis helpers for pandas.

Everything is available from the top level:

    import dstoolkit as dst

    df = dst.convert_types(raw)         # yes/no text -> True/False, date text -> dates
    df = dst.clean(df, missing={"numeric": "median"})
    dst.show_distribution_profile(df["age"])
    dst.profile(df).save("report.html")

or from its module: `dstoolkit.cleaning`, `dstoolkit.exploration` and
`dstoolkit.profiling`.
"""

from . import cleaning, exploration, profiling
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

__version__ = "0.5.0"

__all__ = ["cleaning", "exploration", "profiling", "__version__",
           *cleaning.__all__, *exploration.__all__, *profiling.__all__]
