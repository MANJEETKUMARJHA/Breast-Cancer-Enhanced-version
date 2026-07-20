# inspect_columns.py
import pandas as pd

df = pd.read_csv("data/breast-cancer.csv")
print("Columns in CSV (exact):")
for c in df.columns:
    print(repr(c))
print("\nTotal columns:", len(df.columns))
