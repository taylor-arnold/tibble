from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, List, Mapping, Union, overload

import pandas as pd

from .verbs_columns import drop, rename, select
from .verbs_join import (
    join_anti,
    join_fuzzy,
    join_inner,
    join_left,
    join_outer,
    join_right,
    join_semi,
)
from .verbs_output import to_csv, to_dtm, to_ggplot, to_torch, to_xy
from .verbs_reshape import pivot_longer
from .verbs_rows import arrange, filter, omit_na, slice_head, slice_sample, slice_tail
from .verbs_transform import mutate, summarize, table


@dataclass
class Tibble:
    _df: pd.DataFrame

    def __init__(self, data: Union[pd.DataFrame, Mapping[str, Any]]):
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
        elif isinstance(data, Mapping):
            # Let pandas handle dict -> DataFrame conversion
            self._df = pd.DataFrame(data)
        else:
            raise TypeError(
                f"{self.__class__.__name__} expected a pandas DataFrame or dict-like, "
                f"got {type(data)}"
            )

    # ---------- Constructors / converters ----------

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Tibble":
        return cls(df.copy())

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()

    # ---------- Core dunder methods ----------

    def __repr__(self) -> str:
        return f"{repr(self._df)}"

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return iter(self._df)

    def _repr_html_(self):
        return self._df._repr_html_()

    # ---------- [] access (column selection) ----------

    @overload
    def __getitem__(self, key: Hashable) -> pd.Series: ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> "Tibble": ...

    def __getitem__(self, key: Any) -> Any:
        result = self._df[key]

        if isinstance(result, pd.DataFrame):
            return type(self)(result)

        return result

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._df[key] = value

    # ----------------------- verbs_columns.py  ---------------------------------#
    def select(self, *cols: str | Iterable[str]) -> "Tibble":
        return type(self)(select(self._df, *cols))

    def drop(self, *cols: str | Iterable[str]) -> "Tibble":
        return type(self)(drop(self._df, *cols))

    def rename(self, **new_names) -> "Tibble":
        return type(self)(rename(self._df, **new_names))

    # ----------------------- verbs_rows.py  ------------------------------------#
    def filter(self, fn, groupby=None) -> "Tibble":
        return type(self)(filter(self._df, fn=fn, groupby=groupby))

    def omit_na(self) -> "Tibble":
        return type(self)(omit_na(self._df))

    def arrange(self, *cols: str | Iterable[str]) -> "Tibble":
        return type(self)(arrange(self._df, *cols))

    def slice_head(self, n: int, groupby=None) -> "Tibble":
        return type(self)(slice_head(self._df, n=n, groupby=groupby))

    def slice_tail(self, n: int, groupby=None) -> "Tibble":
        return type(self)(slice_tail(self._df, n=n, groupby=groupby))

    def slice_sample(self, n: int = None, frac: float | None = None, groupby=None) -> "Tibble":
        return type(self)(slice_sample(self._df, n=n, frac=frac, groupby=groupby))

    # ----------------------- verbs_transform.py  -------------------------------#
    def mutate(self, groupby=None, **new_cols) -> "Tibble":
        return type(self)(mutate(self._df, groupby, **new_cols))

    def summarize(self, groupby=None, **metrics) -> "Tibble":
        return type(self)(summarize(self._df, groupby, **metrics))

    def table(self, row: str = None, col: str = None) -> "Tibble":
        return type(self)(table(self._df, row, col))

    # ----------------------- verbs_join.py  ------------------------------------#
    def join_left(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
        suffix: tuple = ("", "_y"),
    ) -> "Tibble":
        return type(self)(join_left(self._df, y._df, on, on_left, on_right, suffix))

    def join_right(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
        suffix: tuple = ("", "_y"),
    ) -> "Tibble":
        return type(self)(join_right(self._df, y._df, on, on_left, on_right, suffix))

    def join_inner(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
        suffix: tuple = ("", "_y"),
    ) -> "Tibble":
        return type(self)(join_inner(self._df, y._df, on, on_left, on_right, suffix))

    def join_outer(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
        suffix: tuple = ("", "_y"),
    ) -> "Tibble":
        return type(self)(join_outer(self._df, y._df, on, on_left, on_right, suffix))

    def join_semi(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
    ) -> "Tibble":
        return type(self)(join_semi(self._df, y._df, on, on_left, on_right))

    def join_anti(
        self,
        y: "Tibble",
        on: str | List[str] | None = None,
        on_left: str | List[str] | None = None,
        on_right: str | List[str] | None = None,
    ) -> "Tibble":
        return type(self)(join_anti(self._df, y._df, on, on_left, on_right))

    def join_fuzzy(
        self,
        y: "Tibble",
        on: str = None,
        on_left: str = None,
        on_right: str = None,
        by: str | List[str] = None,
        by_left: str | List[str] = None,
        by_right: str | List[str] = None,
        suffix: tuple = ("", "_y"),
        direction="nearest",
    ) -> "Tibble":
        return type(self)(
            join_fuzzy(
                self._df,
                y._df,
                on,
                on_left,
                on_right,
                by,
                by_left,
                by_right,
                suffix,
                direction,
            )
        )

    # ----------------------- verbs_reshape.py  ---------------------------------#
    def pivot_longer(
        self,
        id_vars=None,
        value_vars=None,
        names_to="name",
        values_to="value",
    ) -> "Tibble":
        return type(self)(
            pivot_longer(
                self._df,
                id_vars=id_vars,
                value_vars=value_vars,
                names_to=names_to,
                values_to=values_to,
            )
        )

    # ----------------------- verbs_output.py  ----------------------------------#
    def to_ggplot(self, mapping=None):
        return to_ggplot(self._df, mapping=mapping)

    def to_csv(self, path_or_buf) -> None:
        return to_csv(self._df, path_or_buf)

    def to_xy(self, target, features=None, drop=None, as_numpy=True):
        return to_xy(self._df, features=features, drop=drop, as_numpy=as_numpy)

    def to_torch(self, target, features=None, drop=None) -> pd.DataFrame:
        return to_torch(self._df, target, features=features, drop=drop)

    def to_dtm(
        self,
        doc_col: str,
        term_col: str,
        weight_col: str | None = None,
        target_col: str | None = None,
        top_n_terms: int | None = None,
    ) -> pd.DataFrame:
        return to_dtm(self._df, doc_col, term_col, weight_col, target_col, top_n_terms)
