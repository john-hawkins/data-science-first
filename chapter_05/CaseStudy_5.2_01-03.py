import matplotlib.pyplot as plt
import pandas as pd


results = pd.read_csv("data/feature_importance.csv")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].scatter(results['ablation'], results['permutation'], alpha=0.3, s=50)
axes[0].set_xlabel('Ablation Importance')
axes[0].set_ylabel('Permutation Importance')
xticks = axes[0].xaxis.get_major_ticks()
xticks[1].set_visible(False) 
xticks[3].set_visible(False) 
xticks[5].set_visible(False) 
xticks[7].set_visible(False) 
xticks[9].set_visible(False) 

axes[1].scatter(results['ablation'], results['tree'], alpha=0.3, s=50) 
axes[1].set_xlabel('Ablation Importance')
axes[1].set_ylabel('Tree Importance')
xticks = axes[1].xaxis.get_major_ticks()
xticks[1].set_visible(False) 
xticks[3].set_visible(False) 
xticks[5].set_visible(False) 
xticks[7].set_visible(False) 
xticks[9].set_visible(False) 

axes[2].scatter(results['permutation'], results['tree'], alpha=0.3, s=50)   
axes[2].set_xlabel('Permutation Importance')
axes[2].set_ylabel('Tree Importance')
plt.show()


#ab_quantile = results['ablation'].quantile(0.75)
#perm_quantile = results['permutation'].quantile(0.75)

