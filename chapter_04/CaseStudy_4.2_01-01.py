from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd

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
 
X = newdf.loc[:,cols]

k = 7
kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeanModel.labels_
distances_to_all_centroids = kmeanModel.transform(X)

distances_to_assigned_centroid = []
for i, label in enumerate(labels):
    distances_to_assigned_centroid.append(distances_to_all_centroids[i, label])


df['cluster'] = labels
df['dist_to_centroid']=  distances_to_assigned_centroid

newdf = df.loc[:,['source','text', 'generated', 'cluster','dist_to_centroid', 'text_len', 'text_avg_wl']].copy()
newdf.to_csv("data/clustered_data.csv", index=False)

