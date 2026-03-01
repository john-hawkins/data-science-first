import pandas as pd
import json
import re

df = pd.read_csv("data/test_sample_for_text_permutation.csv")

new_rows = []

with open("data/keywords.json") as f:
   lookup = json.load(f)

for index, row in df.iterrows():
   r = row['record_index']
   generated = row['generated']
   source = row['source']
   text = row['text']
   record = {"record":r, "source":source, "text":text, "generated":generated, "word":"NONE", "sub":"NONE"}
   new_rows.append(record)
   for k in lookup.keys():
      sub = lookup[k]
      pattern = r"([^a-zA-Z])" + k + r"([^a-zA-Z])"
      if re.search(pattern, text, re.IGNORECASE):
         replacement = r"\1"+sub+r"\2"
         new_text = re.sub(pattern, replacement, text, re.IGNORECASE)
         new_record = {"record":r, "source":source, "text":new_text, "generated":generated, "word":k, "sub":sub}
         new_rows.append(new_record)

new_df = pd.DataFrame(new_rows)

new_df.to_csv("data/permuted_text_samples.csv", index=False)


