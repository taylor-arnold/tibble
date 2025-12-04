from __future__ import annotations

import pandas as pd


def pivot_longer(
    df: pd.DataFrame,
    id_vars=None,
    value_vars=None,
    names_to="name",
    values_to="value",
) -> pd.DataFrame:
    res = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=names_to,
        value_name=values_to,
    )

    return res


def pivot_wider(df: pd.DataFrame, names_from, values_from=None) -> pd.DataFrame:
    if values_from:
        res = pd.pivot(df, columns=names_from, values=values_from)
    else:
        res = pd.pivot(df, columns=names_from)

    return res
