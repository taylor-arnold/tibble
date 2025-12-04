import numpy as np
import pandas as pd

from tibble import Tibble, concat, read_csv


from numpy import (
    min, max, sum, mean, median, quantile, log, sqrt, exp, floor, abs
)

n_rows = 1000
df = Tibble({
    "id": range(n_rows),
    "category": np.random.choice(["A", "B", "C", "D"], n_rows),
    "category2": np.random.choice(["Z", "Y", "X"], n_rows),
    "value1": np.random.randn(n_rows) * 100,
    "value2": np.random.exponential(scale=50, size=n_rows),
    "value3": np.random.uniform(0, 1, n_rows),
    "constant": "same_value",
    "price": np.random.gamma(2, 50, n_rows),
    "quantity": np.random.poisson(10, n_rows),
})

df2 = Tibble({
    "category": ["A", "B", "E", "D"],
    "quantity": np.random.poisson(10, 4),
})

print(concat([df, df]))

print(df.select("category", "category2"))

print(df.drop("category", "category2"))

print(df.rename(new="value1"))


print(df.filter(lambda d: d["id"] > 500))

print(df.omit_na())

print(df.arrange("category", "-value1"))

print(df.slice_head(n=10))

print(df.slice_head(n=2, groupby="category"))

print(df.slice_tail(n=10))

print(df.slice_tail(n=2, groupby="category"))

print(df.slice_sample(n=2))

print(df.slice_sample(n=2, groupby="category"))


print(df.mutate(new=lambda d: d["value1"] / d["value2"]))

print(df.summarize(new=lambda d: d["value1"].mean(), groupby="category"))

print(df.table("category"))

print(df.table(col="category"))

print(df.table("category", "category2"))


print("\n" + "=" * 45 + "\n[Left Join]")
print(df.join_left(df2, on="category"))

print("\n" + "=" * 45 + "\n[Right Join]")
print(df.join_right(df2, on="category"))

print("\n" + "=" * 45 + "\n[Inner Join]")
print(df.join_inner(df2, on="category"))

print("\n" + "=" * 45 + "\n[Outer Join]")
print(df.join_outer(df2, on="category"))

print("\n" + "=" * 45 + "\n[Semi Join]")
print(df.join_semi(df2, on="category"))

print("\n" + "=" * 45 + "\n[Anti Join]")
#print(df.join_anti(df2, on="category"))

print("\n" + "=" * 45 + "\n[Fuzzy Join]")
print(df.join_fuzzy(df2, on="quantity"))



print(
    df
    .mutate(val1 = "$value1 / $value2")
    .filter("$val1 > 0.5")
    .summarize(mu = "sum($value1)", groupby="category")
)


