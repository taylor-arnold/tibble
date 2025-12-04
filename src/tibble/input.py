from __future__ import annotations

from .tibble import Tibble 

import pandas as pd


def read_csv(*args, **kwargs) -> "Tibble":
    df = pd.read_csv(*args, **kwargs)

    return Tibble(df)
