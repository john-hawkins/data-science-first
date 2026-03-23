import re
import json
import itertools
from tqdm import tqdm
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_list_from_file(file):
    filename = f"data/{file}"
    with open(filename, "r") as f:
        my_list = [line.strip() for line in f.readlines()]
    return my_list


prompt = PromptTemplate(
   template = """
     You are an expert on the English language.
     You will generate a single sentence expression of an idea
     that is consistent with the following criteria.
     ---
     Topic: It should relate to {TOPIC}
     Words: Make use of {CASE}s.
     Style: It should have a {STYLE} style.
     ---
     Generate a single sentence of text, nothing else.
     """,
   input_variables=[ "TOPIC", "CASE", "STYLE"],
)

from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.2")

chain = prompt | model | StrOutputParser()

cases = ['idiom', 'no idiom']
styles = ['colloquial', 'literary']

topics_file = "topics.txt"
topics = get_list_from_file(topics_file)

results = pd.DataFrame()

combinations = list(itertools.product(topics, cases))

for row in tqdm(combinations, desc=f"Processing file in: {topics_file}"):
   t = row[0]
   c = row[1]
   for s in styles:
      record = {"TOPIC": t, "CASE":c, "STYLE":s}
      try:
          response = chain.invoke(record)
          record['text'] = response
       
      except Exception as e:
          record['text'] = "FAILURE" 
      results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

results.to_csv("data/generated_expressions.csv", index_label='ID')


