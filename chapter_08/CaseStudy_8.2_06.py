import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("data/translated_expressions.csv")
fdf = df[df['has_idiom']==True].copy()
fdf['label'] = np.where(fdf['instruction']=="IdiomLiteral", 1, 0)

repeats = fdf['translation'].value_counts().reset_index()
repeats = repeats[repeats['count']>1]

values_to_drop = list(repeats['translation'])
fdf_clean = fdf[~fdf['translation'].isin(values_to_drop)]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
split = gss.split(fdf_clean, groups=fdf_clean['ID'])

train_inds, test_inds = next(split)

train = fdf_clean.iloc[train_inds]
test = fdf_clean.iloc[test_inds]

train.to_csv("data/idiom_classifier_train.csv", index=False)
test.to_csv("data/idiom_classifier_test.csv", index=False)
