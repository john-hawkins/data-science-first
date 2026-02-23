import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier

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

emb_len = len(cols)
train = newdf[newdf["RANDOM"]<0.8]
test = newdf[newdf["RANDOM"]>=0.8]
X_train = train.loc[:,cols]
y_train = train.loc[:,"generated"] 
X_test = test.loc[:,cols]
y_test = test.loc[:,"generated"]

xt = ExtraTreesClassifier()
xt.fit(X_train, y_train)

# Metrics for the Extra Trees Model
y_pred = xt.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = xt.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Length":emb_len, "AUC": auc, "Precision":prec, "Recall":recall}
print(record)

filename = 'xt_BERT_model.pkl'
pickle.dump(xt, open(filename, 'wb'))

