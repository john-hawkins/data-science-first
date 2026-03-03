from datasets import load_dataset
from datasets import get_dataset_split_names

get_dataset_split_names("google/civil_comments")

ds = load_dataset("google/civil_comments", split="train")

cols = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

def make_binary(sample):
    for x in cols:
        if sample[x] > 0:
            sample[x] = 1.0
    return sample

bin_ds = ds.map(make_binary)

## Print out percentages of comment types
for x in cols:
    print( str(round(100*np.mean(bin_ds[x]), 1)) + " % " + x)


def add_proxy(sample):
    scores = [sample[x] for x in cols]
    score = np.sum(scores)
    if score > 0:
        sample['proxy'] = 1.0
    else:
        sample['proxy'] = 0.0
    return sample

updated_ds = ds.map(add_proxy)


from datasets import concatenate_datasets

shuff = updated_ds.shuffle(seed=42)

samples = 100 
pos = shuff.filter(lambda x: x["proxy"]==1).select(range(samples))
neg = shuff.filter(lambda x: x["proxy"]==0).select(range(samples))

case_ds = concatenate_datasets([pos,neg])

