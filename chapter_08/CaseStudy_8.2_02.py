import re
import json
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

model_name = "gemini-2.5-pro"
model = ChatGoogleGenerativeAI(model=model_name)

dataset = "data/generated_expressions_sample_llama3.2.csv"
dataset = "data/generated_expressions.csv"
df = pd.read_csv(dataset)

class Audit(BaseModel):
    has_idiom: bool = Field(
        description="A Boolean value that indicates if the sentence\
                     provided contains an idiom."
    )

parser = JsonOutputParser(pydantic_object=Audit)

prompt = PromptTemplate(
   template = """
     You are an expert on the English language.
     Your job is to determine whether the sentence
     below contains an English idiom.
     ---
     Sentence: {SENTENCE}
     ---
     {format_instructions}
     """,
   input_variables=[ "SENTENCE"],
   partial_variables={
      "format_instructions":parser.get_format_instructions()
   }
)

chain = prompt | model | parser
    
results = pd.DataFrame()

records = [x for i,x in df.iterrows()] 

for r in tqdm(records, desc=f"Processing file in: {dataset}"):

         record = {"SENTENCE": r['text']}
         try:
             response = chain.invoke(record)
             r['has_idiom'] = response['has_idiom']
         except Exception as e:
             print("Error processing record", e)
             r['has_idiom'] = np.nan
         results = pd.concat([results, pd.DataFrame([r])], ignore_index=True)

results.to_csv("data/audited_expressions.csv")


