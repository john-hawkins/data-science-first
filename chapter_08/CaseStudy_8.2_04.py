import pandas as pd

df = pd.read_csv("data/audited_expressions.csv")


filtered = df[ ((df['CASE']=='idiom') & (df['has_idiom']==True)) |
               ((df['CASE']=='no idiom') & (df['has_idiom']==False))
]

filtered.to_csv("data/audited_expressions_filtered.csv", index=False)

filtered_records = len(df)-len(filtered)
pct = round(100*filtered_records/len(df), 2)

print(f"Filtered {pct}% Records down to {len(filtered)} Total Records")

