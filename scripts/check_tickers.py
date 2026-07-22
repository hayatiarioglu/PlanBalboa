import pandas as pd
df = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
print("Unique tickers in dataset:", df["ticker"].unique())
print("Data timestamp range:", df["timestamp"].min(), "to", df["timestamp"].max())
