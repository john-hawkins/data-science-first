import pandas as pd
from matplotlib_venn import venn2
from matplotlib import pyplot as plt

hu_keywords = pd.read_csv("data/human_keywords.csv")
ai_keywords = pd.read_csv("data/ai_keywords.csv")

# Define your sets of keywords
set_ai = set(ai_keywords['keyword'].to_list()) 
set_human = set(hu_keywords['keyword'].to_list()) 

ai_only = (set_ai - set_human)
human_only = (set_human - set_ai)
both_sets = (set_ai & set_human)
 
v = venn2(subsets=(len(ai_only), len(human_only), len(both_sets)), 
      set_labels=('AI Keywords', 'Human Keywords'))

# Modify the content of the circles
if v.get_label_by_id('10'):
    v.get_label_by_id('10').set_text("\n".join(ai_only))

if v.get_label_by_id('01'):
    v.get_label_by_id('01').set_text("\n".join(human_only))

if v.get_label_by_id('11'):
    v.get_label_by_id('11').set_text("\n".join(both_sets))

# Display the plot
plt.title("Keywords for AI and Human Generated Text")
plt.show()

