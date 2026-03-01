from keybert import KeyBERT
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np

df = pd.read_csv("data/complete_with_features.csv")
test = df[df["RANDOM"]>=0.95]

text_list = test['text'].tolist()

model_name = 'bert-base-uncased'
kw_model = KeyBERT(model=model_name)

def extract_keywords(texts, top_n=3):
    all_keywords = []
    for text in tqdm(texts, desc="Extracting keywords"):
        keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
        all_keywords.extend([kw[0] for kw in keywords])
    return Counter(all_keywords)
  
keywds = extract_keywords(text_list, top_n=3)

df_keywords = pd.DataFrame({
    "keywords": pd.Series(keywds)
}).fillna(0).astype(int).reset_index()

df_keywords.columns=['keyword','count']
df_keywords = df_keywords.sort_values(by='count', ascending=False)

keywords = df_keywords[df_keywords['count']>=30]

print(keywords.head(5).to_markdown())
