import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

df = pd.read_csv("data/clustered_data.csv")
tdf = pd.read_csv("data/topic_labels.csv") 

topics = tdf.set_index('ID')['Topic Name'].to_dict()

def topic_lookup(t):
    return topics[t]

df['Topic'] = df['cluster'].apply(topic_lookup)

levels, categories = pd.factorize(df['Topic'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df['text_len'], df['text_avg_wl'], c=colors)
plt.title('Topic Distribution by Document Properties')
plt.xlabel('Document Length')
plt.ylabel('Average Word Length')
plt.legend(handles=handles, title='Topic')
plt.ylim(0, 8)
plt.xlim(0, 10000)

