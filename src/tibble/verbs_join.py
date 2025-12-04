from __future__ import annotations

from typing import List

import pandas as pd


def join_left(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
    suffix: tuple = ("", "_y"),
) -> pd.DataFrame:
    out = pd.merge(
        left,
        right,
        on=on,
        left_on=on_left,
        right_on=on_right,
        suffixes=suffix,
        how="left",
    )

    return out


def join_right(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
    suffix: tuple = ("", "_y"),
) -> pd.DataFrame:
    out = pd.merge(
        left,
        right,
        on=on,
        left_on=on_left,
        right_on=on_right,
        suffixes=suffix,
        how="right",
    )

    return out


def join_inner(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
    suffix: tuple = ("", "_y"),
) -> pd.DataFrame:
    out = pd.merge(
        left,
        right,
        on=on,
        left_on=on_left,
        right_on=on_right,
        suffixes=suffix,
        how="inner",
    )

    return out


def join_outer(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
    suffix: tuple = ("", "_y"),
) -> pd.DataFrame:
    out = pd.merge(
        left,
        right,
        on=on,
        left_on=on_left,
        right_on=on_right,
        suffixes=suffix,
        how="outer",
    )

    return out


def join_semi(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
) -> pd.DataFrame:
    if on is not None and (on_left is not None or on_right is not None):
        raise ValueError("Use either `on` OR (`on_left` and `on_right`), not both.")

    if on is not None:
        right_keys = on
        on_left = on
        on_right = on

    else:
        if on_left is None or on_right is None:
            raise ValueError(
                "When `on` is None, you must provide both `on_left` and `on_right`."
            )
        right_keys = on_right
        merge_kwargs = dict(left_on=on_left, right_on=on_right)

    right_keys_df = right[right_keys].drop_duplicates()

    merged = left.merge(
        right_keys_df,
        how="inner",
        left_on=on_left,
        right_on=on_right
    )

    return merged[left.columns]


def join_anti(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | List[str] | None = None,
    on_left: str | List[str] | None = None,
    on_right: str | List[str] | None = None,
) -> pd.DataFrame:
    if on is not None and (on_left is not None or on_right is not None):
        raise ValueError("Use either `on` OR (`on_left` and `on_right`), not both.")

    if on is not None:
        right_keys = on
        on_right = on
        on_left = on
    else:
        if on_left is None or on_right is None:
            raise ValueError(
                "When `on` is None, you must provide both `on_left` and `on_right`."
            )
        right_keys = on_right

    tmp = left.merge(
        right[right_keys],
        how="left",
        indicator=True,
        on_right=on_right,
        on_left=on_left
    )

    return tmp.loc[tmp["_merge"] == "left_only", left.columns]


def join_fuzzy(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    on_left: str,
    on_right: str,
    by: str | List[str],
    by_left: str | List[str],
    by_right: str | List[str],
    suffix: tuple = ("", "_y"),
    direction="nearest",
) -> pd.DataFrame:
    if on_left:
        if on_left == on_right:
            on = on_left

    if on:
        on_left = on
        on_right = on + suffix[1]
        right = right.rename(columns={on: on_right})
        right = right[[on_right] + [c for c in right.columns if c != on_right]]

    left = left.copy().sort_values(on_left)
    right = right.copy().sort_values(on_right)

    res = pd.merge_asof(
        left=left,
        right=right,
        left_on=on_left,
        right_on=on_right,
        by=by,
        left_by=by_left,
        right_by=by_right,
        suffixes=suffix,
        direction=direction,
    )

    return res
