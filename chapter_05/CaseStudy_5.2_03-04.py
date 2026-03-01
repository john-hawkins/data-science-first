import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/permuted_text_samples.csv")
embeddings = "permutation_embeddings.npy"

def expand_array_col(df, col_name):
    expanded = df[col_name].apply(pd.Series)
    expanded.columns = [f'{col_name}_{i+1}' for i in range(expanded.shape[1])]
    df_expanded = pd.concat([df.drop(col_name, axis=1), expanded], axis=1)
    return df_expanded, expanded.columns

embs = np.load(embeddings, allow_pickle=True,)
df["embedding"] = list(embs)
newdf, cols = expand_array_col(df, "embedding")

refs = newdf[newdf['word']=="NONE"].copy()
subs = newdf[newdf['word']!="NONE"].copy()

diff_cols = ["c_"+x.split("_")[1] for x in cols]

def get_diffs(row, ref_row):
    refs = ref_row[cols]
    vals = row[cols]
    diffs = refs - vals
    return diffs.to_dict()

results = pd.DataFrame()

for index, ref_row in refs.iterrows():
    subset = subs[subs['record']==ref_row['record']].copy()
    new_cols = subset.apply(lambda r: get_diffs(r, ref_row), axis=1, result_type='expand')
    subset[diff_cols] = new_cols
    results = pd.concat([results, subset], ignore_index=True)

results.to_csv("data/permuted_text_samples_with_embeddings.csv", index=False)

words = list(subs['word'].unique())

word_results = []

for w in words:
    subset = results[results['word']==w]
    mean_absolute_values = subset[diff_cols].abs().mean()
    mean_values = subset[diff_cols].mean()
    abs_vals = mean_absolute_values.to_dict()
    mean_vals = mean_values.to_dict()
    abs_c = mean_absolute_values.idxmax()
    mean_c = mean_values.idxmax()
    diffabs = abs_vals[abs_c]    
    diffval = mean_vals[abs_c]    
    record = {"word":w, "component":abs_c, "d_absolute":diffabs, "d_mean":diffval}
    word_results.append(record)    


result_df = pd.DataFrame(word_results)
print(result_df.round(3).to_markdown(index=False))


filename = 'xt_BERT_model.pkl'
xt_model = pickle.load(open(filename, 'rb'))

X_refs = refs.loc[:,cols]
X_subs = subs.loc[:,cols]

scores = xt_model.predict_proba(X_refs)
prob_ai = scores[:,1]
refs["prob_ai"] = prob_ai

scores = xt_model.predict_proba(X_subs)
prob_ai = scores[:,1]
subs["prob_ai"] = prob_ai
 
model_results = pd.DataFrame()

for index, ref_row in refs.iterrows():
    subset = subs[subs['record']==ref_row['record']].copy()
    ref_prob = ref_row["prob_ai"]
    subset["prob_diff"] = subset["prob_ai"] - ref_prob 
    subset["abs_prob_diff"] = np.abs(subset["prob_diff"])
    model_results = pd.concat([model_results, subset], ignore_index=True)

 
agg_func = {"abs_prob_diff":"mean", "prob_diff":"mean"}
grpd = model_results.groupby("word").agg(agg_func).reset_index()
print(grpd.round(3).to_markdown(index=False))
