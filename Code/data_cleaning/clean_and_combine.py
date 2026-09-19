import pandas as pd
import glob

files = glob.glob("raw_data/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file, encoding="latin1")
    df = df.dropna(how="all")
    df.columns = df.columns.str.strip()
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates()
combined = combined.sort_values("tourney_date", ascending=False)
combined = combined.reset_index(drop=True)
combined.to_csv("cleaned_data/clean_matches.csv", index=False)
