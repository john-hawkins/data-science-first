import pandas as pd
import numpy as np

df = pd.read_csv("data/clustered_data.csv")
test = df[df["cluster"]==0]
test = test[test["RANDOM"]>=0.8]

# Filter to records that have at least one keyword
keywords = pd.read_csv("data/keywords.csv")
keywords = keywords['keyword'].to_list()
exp_keywords = ["[^a-zA-Z]" + x + "[^a-zA-Z]" for x in keywords] 
expr = "|".join(exp_keywords)
records = test[test['text'].str.contains(expr, case=False)]

test_ai = records[records["generated"]==1]
test_human = records[records["generated"]==0]

ai_set = test_ai.sample(200, axis=0)
human_set = test_human.sample(200, axis=0)

test_set = pd.concat([ai_set, human_set], ignore_index=True)
test_set['record_index'] = test_set.index
test_set.to_csv("data/test_sample_for_text_permutation.csv", index=False)
