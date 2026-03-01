import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

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

train = newdf[newdf["RANDOM"]<0.8]
test = newdf[newdf["RANDOM"]>=0.8]
X_train = train.loc[:,cols]
y_train = train.loc[:,"generated"] 
X_test = test.loc[:,cols]
y_test = test.loc[:,"generated"]

xt = ExtraTreesClassifier()
xt.fit(X_train, y_train)

# Use the re-trained baseline to create our Tree and Permutation Importances again

tree_imps = xt.feature_importances_
result = permutation_importance(xt, X_test, y_test, n_repeats=10, random_state=42)
perm_imps =result.importances_mean

# Baseline Metrics for the Extra Trees Model
scores = xt.predict_proba(X_test)
baseline = roc_auc_score(y_test, scores[:,1])
 
diffs = []

for col in cols:
    temp_cols = [x for x in cols if x!=col]
    X_train = train.loc[:,temp_cols]
    X_test = test.loc[:,temp_cols]
    xt = ExtraTreesClassifier()
    xt.fit(X_train, y_train)
    scores = xt.predict_proba(X_test)
    auc = roc_auc_score(y_test, scores[:,1])
    diff = baseline - auc
    diffs.append(diff)


results = pd.DataFrame(
   {
      "feature":cols,
      "permutation":perm_imps,
      "tree":tree_imps,
      "ablation":diffs
   }
)

results.to_csv("data/feature_importance.csv", index=False)
