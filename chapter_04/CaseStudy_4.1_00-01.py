import pandas as pd
import numpy as np
import random

df1 = pd.read_csv("data/llm-detect-ai-generated-text/train_essays.csv")
df2 = pd.read_csv("data/train_v2_drcat_02.csv")
df3 = pd.read_csv("data/Training_Essay_Data.csv")

df1["source"] = "LA-Lab"
df2["source"] = "Darek"
df3["source"] = "Sunil"

df2["generated"] = df2["label"]

cols = ["source", "text", "generated"]
df1 = df1.loc[:,cols] 
df2 = df2.loc[:,cols] 
df3 = df3.loc[:,cols] 

df = pd.concat([df1, df2, df3], ignore_index=True)
records = len(df)

# Drop duplicates
df.drop_duplicates(subset=['text'], keep='first', inplace=True, ignore_index=True)

new_records = len(df)
print("Dropped", records-new_records, "Records")

df['RANDOM'] = df.apply(lambda x: random.random(), axis=1)
df.to_csv("data/complete_dataset.csv", index=False)

