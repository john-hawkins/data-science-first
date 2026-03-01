from keybert import KeyBERT
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np

df = pd.read_csv("data/clustered_data.csv")

test = df[df["cluster"]==0]
test = test[test["RANDOM"]>=0.8]

test_ai = test[test["generated"]==1]
test_human = test[test["generated"]==0]

text_ai = test_ai['text'].tolist()
text_human = test_human['text'].tolist()

model_name = 'bert-base-uncased'
kw_model = KeyBERT(model=model_name)

def extract_keywords(texts, top_n=3):
    all_keywords = []
    for text in tqdm(texts, desc="Extracting keywords"):
        keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
        all_keywords.extend([kw[0] for kw in keywords])
    return Counter(all_keywords)

keywds_ai = extract_keywords(text_ai, top_n=3)
keywds_human = extract_keywords(text_human, top_n=3)

ai_keywords = pd.DataFrame({"keywords": pd.Series(keywds_ai)}).fillna(0).astype(int).reset_index()
ai_keywords.columns=['keyword','count']
ai_keywords = ai_keywords[ai_keywords['count']>=15]
ai_keywords.to_csv("data/ai_keywords.csv", index=False)

hu_keywords = pd.DataFrame({"keywords": pd.Series(keywds_human)}).fillna(0).astype(int).reset_index()
hu_keywords.columns=['keyword','count']
hu_keywords = hu_keywords[hu_keywords['count']>=45]
hu_keywords.to_csv("data/human_keywords.csv", index=False)

