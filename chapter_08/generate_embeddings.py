from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from tqdm  import tqdm
import argparse

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

task = """
   Classify the given customer email as belonging to one of the following categories: 
   Accounts: complaints about accounts and billing
   Network: Complains about the network services
   Hardware: Complaints about hardware failures
   Other: Any other complaint about the company or something else.
   """

def embed_text(text):
    input = get_detailed_instruct(task,text)
    embedding = model.encode(input)
    return embedding


def main(input_path, text_col, output_path):
    df = pd.read_csv(input_path)
    embeddings = []
    rows = [x for i,x in df.iterrows()]
    for row in tqdm(rows, desc=f"Processing: {input_path}"):
        try:
           embeddings.append(embed_text(row[text_col]))
        except Exception as e:
           print(e)

    np.save(output_path, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataframe into Instruct embeddings.")
    parser.add_argument('in_dir', type=str, help='Path to CSV input.')
    parser.add_argument('text_col', type=str, help='name of the text column.')
    parser.add_argument('out_file', type=str, help='Path to output file (numpy array).')
    args = parser.parse_args()
    main(args.in_dir, args. text_col, args.out_file)

