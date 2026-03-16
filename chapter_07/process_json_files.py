from tqdm import tqdm
import pandas as pd
import argparse
import json
import os

import naics


def extract_json_data(file_path):
    result = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            result['uuid'] = data['uuid'] 
            result['language'] = data['language'] 
            result['title'] = data['title'] 
            result['text'] = data['text'] 
            result['categories'] = "|".join(data['categories'])
            temp = naics.categorize_text(result['text'])
            result.update(temp)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check if the file contains valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return result


def main(dir_path, output_path):
    results = pd.DataFrame()
    #for entry in os.scandir(dir_path):
    contents = list(os.scandir(dir_path))
    for entry in tqdm(contents, desc=f"Processing JSON files in: {dir_path}"):
        if entry.is_file():
            if entry.name.endswith(".json"):
                record = extract_json_data(entry.path)
                results = pd.concat([results,pd.DataFrame([record])], ignore_index=True)
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of web.io JSON files.")
    parser.add_argument('in_dir', type=str, help='Path to data.')
    parser.add_argument('out_file', type=str, help='The output file.')
    args = parser.parse_args()
    main(args.in_dir, args.out_file)

