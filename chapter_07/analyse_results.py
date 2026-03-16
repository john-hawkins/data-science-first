import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

def main(file_one, file_two):
    df1 = pd.read_csv(file_one, dtype=str, na_values="")
    df2 = pd.read_csv(file_two, dtype=str, na_values="")
    df1_unknowns = len(df1[df1['lvl2']=="00"])
    df2_unknowns = len(df2[df2['lvl2']=="00"])
    print("## UNKNOWNS")
    print(file_one, "\t:", str(round(100*df1_unknowns/len(df1),2)), "%")
    print(file_two, "\t:", str(round(100*df2_unknowns/len(df2),2)), "%") 

    df1_sorted = df1.sort_values(by='lvl2')
    df2_sorted = df2.sort_values(by='lvl2')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6));
    sns.countplot(x='lvl2', ax=axes[0], data=df1_sorted, palette='viridis', hue='lvl2', legend=False)
    axes[0].set_title('NAICS Categories in File One')
    axes[0].set_xlabel('Lvl2 Category')
    axes[0].set_ylabel('Frequency')
    sns.countplot(x='lvl2', ax=axes[1], data=df2_sorted, palette='viridis', hue='lvl2', legend=False)
    axes[1].set_title('NAICS Categories in File Two')
    axes[1].set_xlabel('Lvl2 Category')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("results/case_study_7.1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse pair of files.")
    parser.add_argument('file_one', type=str, help='Path to data.')
    parser.add_argument('file_two', type=str, help='The output file.')
    args = parser.parse_args()
    main(args.file_one, args.file_two)


