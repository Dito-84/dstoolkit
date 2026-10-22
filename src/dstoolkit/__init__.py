"""dstoolkit: data cleaning and exploratory analysis helpers for pandas.

Everything is available from the top level:

    import dstoolkit as dst

    df = dst.clean(raw, missing={"numeric": "median"})
    dst.show_distribution_profile(df["age"])
    dst.profile(df).save("report.html")

or from its module: `dstoolkit.cleaning`, `dstoolkit.exploration` and
`dstoolkit.profiling`.
"""

from . import cleaning, exploration, profiling
from .cleaning import (
    add_age_column,
    clean,
    encode_categorical,
    fill_missing,
    find_outliers,
    handle_outliers,
    log_transform,
    normalize,
    remove_duplicates,
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
    show_histogram,
)

__version__ = "0.4.1"

__all__ = ["cleaning", "exploration", "profiling", "__version__",
           *cleaning.__all__, *exploration.__all__, *profiling.__all__]
