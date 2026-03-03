import pandas as pd
import argparse
import fitz
import docx
import os

def extract_text_from_docx(filename):
    doc = docx.Document(filename)
    text_list = []
    for para in doc.paragraphs:
        text_list.append(para.text)
    return '\n'.join(text_list)


def extract_text_from_pdf(filename):
    with fitz.open(filename) as doc:
        text_list = []
        for page in doc:
            #text_list.append(page.get_text())
            text_list.append(page.get_textpage().extractText())
        return '\n'.join(text_list)


def main(dir_path, output_path):
    results = pd.DataFrame()
    for entry in os.scandir(dir_path):
        text = ""
        if entry.is_file():
            if entry.name.endswith(".docx"):
                text = extract_text_from_docx(entry.path)
            if entry.name.endswith(".pdf"):
                text = extract_text_from_pdf(entry.path)
        if text != "":
            record = {"file":entry.name, "text":text}
            results = pd.concat([results,pd.DataFrame([record])], ignore_index=True)
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of resume.")
    parser.add_argument('in_dir', type=str, help='Path to resumes.')
    parser.add_argument('out_file', type=str, help='The output file.')
    args = parser.parse_args()
    main(args.in_dir, args.out_file)


