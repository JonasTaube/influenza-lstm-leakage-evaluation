import pandas as pd
df = pd.read_csv(r"../tables/main_series_2015_2025.csv")
print(df.columns.tolist())
print("total weeks:", len(df))
INPUT_LEN, OUTPUT_LEN = 12, 4
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * 0.7)
print("windows:", n_win, "train:", split, "test:", n_win - split)
# first test target week index
first_te = INPUT_LEN + split
print("first test target row:", first_te)
print(df.iloc[[first_te, first_te+1, len(df)-1]].to_string())
