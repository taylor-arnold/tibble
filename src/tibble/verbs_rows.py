from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import utils


def filter(df: pd.DataFrame, fn, groupby=None) -> pd.DataFrame:
    if isinstance(fn, str):
        fn = utils.compile_expr(fn)

    grouped = utils.make_groups(df, groupby, to_iter=True)

    out = []
    for _, group_df in grouped:
        group_df = group_df.reset_index(drop=True).copy()
        group_df = group_df.pipe(lambda g: g[fn(g)])
        out.append(group_df)

    return pd.concat(out)


def omit_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="any")


def arrange(df: pd.DataFrame, *cols: str | Iterable[str]) -> pd.DataFrame:
    norm_cols = utils.normalize_columns_args(*cols)

    ascending = [not c.startswith("-") for c in norm_cols]
    norm_cols = [c.lstrip("-") for c in norm_cols]

    return df.sort_values(norm_cols, ascending=ascending).reset_index(drop=True)


def slice_head(df: pd.DataFrame, n: int, groupby=None) -> pd.DataFrame:
    grouped = utils.make_groups(df, groupby)
    return grouped.head(n).reset_index(drop=True)


def slice_tail(df: pd.DataFrame, n: int, groupby=None) -> pd.DataFrame:
    grouped = utils.make_groups(df, groupby)
    return grouped.tail(n).reset_index(drop=True)


def slice_sample(
    df: pd.DataFrame, n: int | None = None, frac: float | None = None, groupby=None
) -> pd.DataFrame:
    grouped = utils.make_groups(df, groupby)
    return grouped.sample(n=n, frac=frac).reset_index(drop=True)
