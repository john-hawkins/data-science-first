import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/complete_with_features.csv")
embeddings = "Embeddings_bert-base-uncased.npy"

def expand_array_col(df, col_name):
    expanded = df[col_name].apply(pd.Series)
    expanded.columns = [f'{col_name}_{i+1}' for i in range(expanded.shape[1])]
    df_expanded = pd.concat([df.drop(col_name, axis=1), expanded], axis=1)
    return df_expanded, expanded.columns

embs = np.load(embeddings, allow_pickle=True,)
df["embedding"] = list(embs)
newdf, cols = expand_array_col(df, "embedding")

test = newdf[newdf["RANDOM"]>=0.8]
X_test = test.loc[:,cols]
y_test = test.loc[:,"generated"]

filename = 'xt_BERT_model.pkl'
xt_model = pickle.load(open(filename, 'rb'))

tree_imps = xt_model.feature_importances_

result = permutation_importance(xt_model, X_test, y_test, n_repeats=10, random_state=42)
perm_imps =result.importances_mean

results = pd.DataFrame({"feature":cols, "permutation":perm_imps, "decomposition":tree_imps})

print(result.importances_mean)

 
