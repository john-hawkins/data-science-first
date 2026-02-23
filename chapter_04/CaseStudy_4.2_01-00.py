
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

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

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
sil = []

K = range(2, 28)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X_train)
    distortions.append(sum(np.min(cdist(X_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X_train.shape[0])
    inertias.append(kmeanModel.inertia_)
    labels = kmeanModel.labels_
    sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

print("Inertia values:")
for key, val in mapping2.items():
    print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.axvline(x=7, color='red')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

print("Silouette values:")
plt.plot(K, sil, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Silouette Method')
plt.show()


