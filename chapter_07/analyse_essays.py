import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import pandas as pd
import argparse
import os

def main(file_one):
    df1 = pd.read_csv(file_one) # , dtype=str, na_values="")
    print("Essay Features Analysis")
    print(" Detailed vocab\t:", str(round(100* len(df1[df1['detailed_vocab']>0])/len(df1),2)), "%")
    print("     Rare vocab\t:", str(round(100* len(df1[df1['rare_vocab']>0])/len(df1),2)), "%")
    print(" Literary vocab\t:", str(round(100* len(df1[df1['literary_vocab']>0])/len(df1),2)), "%")
    print()
    readability_metrics = [
        "flesch_reading_ease", "flesch_kincaid_grade_level", "automated_readability_index",
        "gunning_fog_index", "smog_readability"
    ]
    readability_metric = ""
    readability_correl = 0
    read_zeros = 0

    vocabulary_metrics = [
        "type_token_ratio", "moving_average_ttr", "yule_k_characteristic",
        "average_word_length", "syllable_complexity"
    ]
    vocabulary_metric = ""
    vocabulary_correl = 0
    vocab_zeros = 0

    for rm in readability_metrics:
         temp = scipy.stats.pearsonr(df1[rm], df1['readability'])
         if temp[0] > readability_correl:
             readability_correl = temp[0]
             readability_metric = rm
         if temp[0] < 0:
             read_zeros += 1
 
    print(f"Best Correlated Readability Metric {readability_metric}")
    print(f" - Correlation : {str( round(readability_correl,2))}")
    print(f" - Proportion of negative correlations: {str(round(100*read_zeros/len(readability_metrics),2))}") 
    print()
    for vm in vocabulary_metrics:
         temp = scipy.stats.pearsonr(df1[vm], df1['vocabulary'])
         if temp[0] > vocabulary_correl:
             vocabulary_correl = temp[0]
             vocabulary_metric= vm
         if temp[0] < 0:
             vocab_zeros += 1

    print(f"Best Correlated Vocabulary Metric {vocabulary_metric}")
    print(f" - Correlation : {str(round(vocabulary_correl,2))}")
    print(f" - Proportion of negative correlations: {str(round(100*vocab_zeros/len(vocabulary_metrics),2))}") 
    print()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6));
    sns.scatterplot(x='readability', y=readability_metric, ax=axes[0], data=df1, legend=False)
    axes[0].set_title('Readability Metrics')
    axes[0].set_xlabel("readability")
    axes[0].set_ylabel(readability_metric)
    sns.scatterplot(x='vocabulary', y=vocabulary_metric, ax=axes[1], data=df1, legend=False)
    axes[1].set_title('Vocabulary Metrics')
    axes[1].set_xlabel('vocabulary')
    axes[1].set_ylabel(vocabulary_metric)
    plt.tight_layout()
    plt.savefig("results/case_study_7.2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a file of scored essays.")
    parser.add_argument('file_one', type=str, help='Path to data.')
    args = parser.parse_args()
    main(args.file_one)


