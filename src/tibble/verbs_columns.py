from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import utils


def select(df: pd.DataFrame, *cols: str | Iterable[str]) -> pd.DataFrame:
    normalized_cols = utils.normalize_columns_args(*cols)

    missing = [c for c in normalized_cols if c not in df.columns]
    if missing:
        raise KeyError(f"select: columns not found: {missing}")

    out = df.loc[:, list(normalized_cols)].copy()
    out = out.reset_index(drop=True)
    return out


def drop(df: pd.DataFrame, *cols: str | Iterable[str]) -> pd.DataFrame:
    normalized_cols = utils.normalize_columns_args(*cols)

    missing = [c for c in normalized_cols if c not in df.columns]
    if missing:
        raise KeyError(f"drop: columns not found: {missing}")

    out = df.drop(columns=list(normalized_cols)).copy()
    out = out.reset_index(drop=True)
    return out


def rename(df: pd.DataFrame, **new_names) -> pd.DataFrame:
    old_cols = list(new_names.values())
    missing = [c for c in old_cols if c not in df.columns]

    if missing:
        raise KeyError(f"rename: columns not found: {missing}")

    rename_map = {old: new for new, old in new_names.items()}

    out = df.rename(columns=rename_map).copy()
    out = out.reset_index(drop=True)
    return out
