"""
Simple visualisation of the semantic separation among the generated emails.
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches
import pandas as pd
import numpy as np

datafile = "data/generated_complaint_emails_final.csv"
embeddings = "data/complaint_email_embeddings_final_Instruct.npy"

df = pd.read_csv(datafile)

def expand_array_col(df, col_name):
    expanded = df[col_name].apply(pd.Series)
    expanded.columns = [f'{col_name}_{i+1}' for i in range(expanded.shape[1])]
    df_expanded = pd.concat([df.drop(col_name, axis=1), expanded], axis=1)
    return df_expanded, expanded.columns

embs = np.load(embeddings, allow_pickle=True,)
df["embedding"] = list(embs)
newdf, cols = expand_array_col(df, "embedding")
 
X = newdf.loc[:,cols]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
 
pca = PCA(n_components=3)
pca.fit(scaled_data)

transformed_data = pca.transform(scaled_data)

df['dim1'] = transformed_data[:,0]
df['dim2']=  transformed_data[:,1]
df['dim3']=  transformed_data[:,2]


emotions = list(df['EMOTION'].unique())
edus = list(df['EDU'].unique())
cats = list(df['CATEGORY'].unique())
markers = ['v', '^', '*', 'x', 'd', 'o', '+']

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(projection='3d')

for c, group_data in df.groupby('CATEGORY'):
    print(f"Category {c} has {len(group_data)} records")
    ax.scatter(group_data['dim1'], group_data['dim2'], group_data['dim3'],
         marker=markers[cats.index(c)], alpha=0.3,
         label=c, s=50)  

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

ax.legend(title='Complaint Category')
plt.show()


######

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for c, group_data in df.groupby('CATEGORY'):
    print(f"Category {c} has {len(group_data)} records")
    axes[0].scatter(group_data['dim1'], group_data['dim2'],
         marker=markers[cats.index(c)], alpha=0.3,
         label=c, s=50) # s for marker size

axes[0].legend(title='Complaint Category')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

for e, group_data in df.groupby('EDU'):
    axes[1].scatter(group_data['dim1'], group_data['dim2'],
               marker=markers[edus.index(e)], alpha=0.3,
               label=e, s=50) # s for marker size

axes[1].legend(title='Education Level')
axes[1].set_xlabel('Principal Component 1')
 
for e, group_data in df.groupby('EMOTION'):
    axes[2].scatter(group_data['dim1'], group_data['dim2'],
               marker=markers[emotions.index(e)], alpha=0.3,
               label=e, s=50) # s for marker size

axes[2].legend(title='Education Level')
axes[2].set_xlabel('Principal Component 1')

plt.show()

