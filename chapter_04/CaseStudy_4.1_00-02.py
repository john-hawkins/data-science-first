import pandas as pd
import numpy as np

df = pd.read_csv("data/complete_dataset.csv")

df["chars"] = df['text'].apply(len)

def word_count(t):
    wds = t.split(" ")
    return len(wds)

df["words"] = df['text'].apply(word_count)

def word_len(t):
    wds = t.split(" ")
    lens = [len(x) for x in wds]
    return np.mean(lens)

df["avg_wd"] = df.apply(lambda x: word_len(x['text']), axis='columns')

df["creator"] = np.where(df['generated']==1,"GenAI","Human")

summary = df.groupby(["source","creator"]).agg({"generated":"count","chars":"mean", "words":"mean","avg_wd":"mean"}).reset_index()
summary = summary.round(1)

summary.columns = ["Data", "Origin", "Records", "Avg Chrs", "Avg Wds", "Avg Wd Len"]


# Display DataFrame as a Markdown Table
markdown_table = summary.to_markdown(index=False)
print(markdown_table)




