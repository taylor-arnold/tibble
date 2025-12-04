from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Sequence, List

import pandas as pd


def normalize_columns_args(*cols) -> Sequence[str]:
    if (
        len(cols) == 1
        and isinstance(cols[0], Iterable)
        and not isinstance(cols[0], (str, bytes))
    ):
        return list(cols[0])
    else:
        return list(cols)


def make_groups(
    df: pd.DataFrame, by_: str | Sequence[str] | None = None, to_iter=False
) -> List:
    if by_ is None:
        if to_iter:
            return [(None, df)]
        else:
            return df
    elif isinstance(by_, str):
        group_cols = [by_]
    else:
        group_cols = list(by_)

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise KeyError(f"grouping columns not found: {missing_group_cols}")

    grouped = df.copy().groupby(group_cols)

    return grouped


def compile_expr(expr: str, caller_globals=None):
    """
    Turn an expression like "$a + np.mean($b)" into a function:
        f(d) -> d["a"] + np.mean(d["b"])
    where `d` is the DataFrame.
    """
    rewritten = re.sub(r"\$([A-Za-z_]\w*)", r'd["\1"]', expr)

    code = compile(rewritten, "<mutate-expr>", "eval")

    # Use caller's globals if provided, otherwise use this module's globals
    eval_globals = caller_globals if caller_globals is not None else globals()

    def fn(d, _code=code, _globals=eval_globals):
        return eval(_code, _globals, {"d": d})

    return fn
