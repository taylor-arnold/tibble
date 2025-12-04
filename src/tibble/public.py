from __future__ import annotations

from .tibble import Tibble 

import numpy as np
import pandas as pd


def concat(objs, **kwargs) -> "Tibble":
    df = Tibble(pd.concat([x._df for x in objs], **kwargs))
    return df


def notin(element, test_elements):
    return np.isin(element, test_elements, invert=True)


def isin(element, test_elements):
    return np.isin(element, test_elements, invert=True)


def isna(obj):
    return pd.isna(obj)


def notna(obj):
    return pd.notna(obj)


def nth(arr, n, default=None):
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError("nth only supports 1D arrays")

    n = int(n)
    L = a.shape[0]

    idx = n if n >= 0 else L + n

    if 0 <= idx < L:
        return a[idx]

    return default


def first(arr, default=None):
    return nth(arr, 0, default=default)


def last(arr, default=None):
    return nth(arr, -1, default=default)


def _infer_window_default(arr, default):
    if default is not None:
        return default

    dt = np.asarray(arr).dtype

    if np.issubdtype(dt, np.floating):
        return np.nan
    if np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.bytes_):
        return ''
    if np.issubdtype(dt, np.bool_):
        return False

    return 0


def lead(arr, n=1, default=None):
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError("only supports 1D arrays")

    n = int(n)
    if n == 0:
        return a.copy()

    L = a.shape[0]
    default = _infer_default(a, default)

    out = np.empty_like(a)
    out[...] = default

    if n > 0:
        if n < L:
            out[:-n] = a[n:]
    else:
        k = -n
        if k < L:
            out[k:] = a[:-k]

    return out


def lag(arr, n=1, default=None):
    return lead(arr, -n, default=default)
