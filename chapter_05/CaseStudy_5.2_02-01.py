import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
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

test = newdf[newdf["RANDOM"]>=0.8]
X_test = test.loc[:,cols]
y_test = test.loc[:,"generated"]

filename = 'xt_BERT_model.pkl'
xt_model = pickle.load(open(filename, 'rb'))

explainer = shap.TreeExplainer(xt_model)

shap_values = explainer.shap_values(X_test)

feat_names = list(cols)
shap_values_0 = shap_values[:,:,0]
shap_values_1 = shap_values[:,:,1]
shap.summary_plot(shap_values_1, X_test, feature_names=feat_names, show=False)
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.show()


### ALTERNATIVE
## VIOLIN PLOTS SIDE BY SIDE
plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
shap.plots.violin(shap_values_0, feature_names=feat_names, show=False)
plt.set_title('SHAP Violin Plot For Human Text')
plt.subplot(1,2,2)
shap.plots.violin(shap_values_1, feature_names=feat_names, color="red", show=False)
plt.set_title('SHAP Violin Plot For AI Text')
plt.show()

