import pandas as pd
df = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
tuprs = df[df["ticker"] == "TUPRS"].sort_values("timestamp")
print("TUPRS max timestamp in dataset:", tuprs["timestamp"].max())
print("TUPRS last 5 rows:\n", tuprs[["timestamp", "close"]].tail(5))
