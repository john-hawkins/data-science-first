import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# SUPPORT MODULES 
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/complete_with_features.csv")

embeddings = {
    "BERT-Mean":"Embeddings_bert-base-uncased.npy",
    "BERT-CLS":"Embeddings_CLS_bert-base-uncased.npy",
    "TaylorAI":"Embeddings_TaylorAI.npy",
    "MiniLm":"Embeddings_MiniLm.npy",
    "Instruct":"Embeddings_instruct.npy",
}

def expand_array_col(df, col_name):
    expanded = df[col_name].apply(pd.Series)
    expanded.columns = [f'{col_name}_{i+1}' for i in range(expanded.shape[1])]
    df_expanded = pd.concat([df.drop(col_name, axis=1), expanded], axis=1)
    return df_expanded, expanded.columns

results = pd.DataFrame(columns=["Embedding", "Length", "AUC", "Precision", "Recall"])

for k in embeddings.keys():
    print(f"Loading : {k}")
    embs = np.load(embeddings[k], allow_pickle=True,)
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
    record = {"Embedding":k,"Length":emb_len, "AUC": auc, "Precision":prec, "Recall":recall}
    print(record)
    results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

results = results.round(3)

# Display Results DataFrame as a Markdown Table
markdown_table = results.to_markdown(index=False)
print(markdown_table)

