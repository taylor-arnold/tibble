from __future__ import annotations

import pandas as pd
import plotnine
import torch
from scipy.sparse import csr_matrix


def to_ggplot(df: pd.DataFrame, mapping=None) -> pd.DataFrame:
    return plotnine.ggplot(df, mapping)


def to_csv(df: pd.DataFrame, path_or_buf) -> None:
    return df.to_csv(path_or_buf, index=False)


def to_xy(df: pd.DataFrame, target, features=None, drop=None, as_numpy=True):
    df = df.copy()
    y = df[target].to_numpy()

    if features:
        X = df[[features]]
    elif drop:
        X = df.drop(columns=drop)
    else:
        X = df.drop(columns=[target])

    if as_numpy:
        X = X.to_numpy()

    return X, y


def to_torch(df: pd.DataFrame, target, features=None, drop=None):
    X, y = to_xy(df, target, features=None, drop=None)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    return X_t, y_t


def to_dtm(
    df: pd.DataFrame,
    doc_col: str,
    term_col: str,
    weight_col: str | None = None,
    target_col: str | None = None,
    top_n_terms: int | None = None,
) -> pd.DataFrame:
    if weight_col is None:
        df_agg = df.groupby([doc_col, term_col]).size().reset_index(name="__weight__")
    else:
        df_agg = (
            df.groupby([doc_col, term_col], as_index=False)[weight_col]
            .sum()
            .rename(columns={weight_col: "__weight__"})
        )

    if df_agg.empty:
        empty_mat = csr_matrix((0, 0))
        empty_docs = pd.Index([], name=doc_col).to_numpy()
        empty_terms = pd.Index([], name=term_col).to_numpy()
        y = None
        if target_col is not None:
            import numpy as np

            y = np.array([])
        return empty_mat, empty_docs, empty_terms, y

    if top_n_terms is not None:
        term_totals = (
            df_agg.groupby(term_col)["__weight__"].sum().sort_values(ascending=False)
        )
        top_terms = term_totals.head(top_n_terms).index
        df_agg = df_agg[df_agg[term_col].isin(top_terms)]

        if df_agg.empty:
            empty_mat = csr_matrix((0, 0))
            empty_docs = pd.Index([], name=doc_col).to_numpy()
            empty_terms = pd.Index([], name=term_col).to_numpy()
            y = None
            if target_col is not None:
                import numpy as np

                y = np.array([])
            return empty_mat, empty_docs, empty_terms, y

    doc_index = pd.Index(sorted(df_agg[doc_col].unique()), name=doc_col)
    term_index = pd.Index(sorted(df_agg[term_col].unique()), name=term_col)

    doc_to_row = {doc: i for i, doc in enumerate(doc_index)}
    term_to_col = {term: j for j, term in enumerate(term_index)}

    row_ids = df_agg[doc_col].map(doc_to_row).to_numpy()
    col_ids = df_agg[term_col].map(term_to_col).to_numpy()
    data = df_agg["__weight__"].to_numpy()

    mat = csr_matrix(
        (data, (row_ids, col_ids)), shape=(len(doc_index), len(term_index))
    )

    y = None
    if target_col is not None:
        doc_targets = (
            df[[doc_col, target_col]]
            .drop_duplicates(subset=[doc_col])  # keeps first occurrence
            .set_index(doc_col)[target_col]
        )
        y = doc_targets.reindex(doc_index).to_numpy()

    return mat, doc_index.to_numpy(), term_index.to_numpy(), y
