from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from . import utils


def mutate(
    df: pd.DataFrame,
    groupby: str | Sequence[str] | None = None,
    **new_cols: Any,
) -> pd.DataFrame:
    grouped = utils.make_groups(df, groupby, to_iter=True)

    out = []
    for key, group_df in grouped:
        group_df = group_df.reset_index(drop=True).copy()
        for name, fn in new_cols.items():
            if isinstance(fn, str):
                fn = utils.compile_expr(fn)

            group_df[name] = fn(group_df)

        out.append(group_df)

    return pd.concat(out)


def summarize(
    df: pd.DataFrame,
    groupby: str | Sequence[str] | None = None,
    **metrics: Any,
) -> pd.DataFrame:
    for key, value in metrics.items():
        if isinstance(value, str):
            metrics[key] = utils.compile_expr(value)

    metrics_series = pd.Series(metrics)

    if not groupby:
        out = metrics_series.apply(lambda f: f(df)).to_frame().T
        return out

    grouped = utils.make_groups(df, groupby, to_iter=True)
    group_cols = grouped.keys

    out = []
    for key, group_df in grouped:
        group_df = group_df.reset_index(drop=True).copy()
        if not isinstance(key, tuple):
            key = (key,)
        key_df = pd.DataFrame([dict(zip(group_cols, key))])
        metrics_df = metrics_series.apply(lambda f: f(group_df)).to_frame().T
        out.append(key_df.join(metrics_df))

    return pd.concat(out)


def table(df: pd.DataFrame, row: str = None, col: str = None):
    if row and col:
        res = pd.crosstab(df[row], df[col])
        res = res.reset_index(names=row)
        res.columns.name = None
    elif row:
        res = df.value_counts(row).reset_index()
    elif col:
        res = df.value_counts(col).to_frame().T
        res.columns.name = None
    else:
        raise TypeError("Must supply at least on ot 'col' or 'row' to table().")

    return res
