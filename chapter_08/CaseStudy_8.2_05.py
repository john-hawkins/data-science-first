import re
import json
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="qwen3:30b")

target_language = "Spanish"

instructions = {
   "idiom":{
      "IdiomLiteral": "Ignore the idiom in this sentence and translate\
                       it perfectly literally",
      "idiomPositive": f"Identify the idiom in this sentence and translate\
                       it into semantically equivalent {target_language}"
   },
   "no idiom":{
      "NoIdiomLiteral": f"Translate this sentence into {target_language}\
                          without using idioms",
      "NoIdiomPositive": f"Translate this sentence into {target_language}\
                          using an appropriate idiom",
   }
}



prompt = PromptTemplate(
   template = """
     You are an expert translator working between English and {TARGET}.
     We give you a sentence in English below which you should translate
     into {TARGET}. You should use the additional instruction below to
     determine how you do the translation.
     ---
     Instruction: {INSTRUCTION}
     Sentence: {SENTENCE}
     ___
     Generate a single sentence transation, nothing else
     """,
   input_variables=["TARGET", "INSTRUCTION", "SENTENCE"],
)

chain = prompt | model | StrOutputParser()

dataset = "data/audited_expressions_filtered.csv"
df = pd.read_csv(dataset)
    
results = pd.DataFrame()

records = [x for i,x in df.iterrows()] 

for r in tqdm(records, desc=f"Processing file in: {dataset}"):

         cased = r['CASE']
         instr = instructions[cased]
         for ins in instr.keys():
             record = {"TARGET": target_language, "SENTENCE": r['text']}
             record["INSTRUCTION"] = instr[ins] 
             try:
                 response = chain.invoke(record)
                 r['instruction'] = ins
                 r['translation'] = response
             except Exception as e:
                 print("Error processing record", e)
                 r['instruction'] = ins
                 r['translation'] = np.nan
             results = pd.concat([results, pd.DataFrame([r])], ignore_index=True)

results.to_csv("data/translated_expressions.csv")


