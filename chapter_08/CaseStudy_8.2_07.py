import re
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

model_name = "gemini-2.5-pro"
model = ChatGoogleGenerativeAI(model=model_name)

class Translation(BaseModel):
    literal_only: bool = Field(
        description="A Boolean value that indicates if the translation is\
                     an erroneous literal translation of the sentence. This\
                     value should be True only if the translation ignores an\
                     idiom in the sentence and translates it directly."
    )

parser = JsonOutputParser(pydantic_object=Translation)

examples = """
English: Don't break the bank, but you can still paint the town red on a shoestring budget with a little creativity and planning!
Spanish: No rompas el banco, pero aún puedes pintar la ciudad de rojo con un presupuesto mínimo con un poco de creatividad y planificación.
literal_only: True
---
English: Don't break the bank, but you can still paint the town red on a shoestring budget with a little creativity and planning!
Spanish: No gastes de más, pero aún puedes pintar la ciudad de rojo con un presupuesto ajustado gracias a un poco de creatividad y planificación.
literal_only: False

"""

prompt = PromptTemplate(
   template = """
     You are an expert on translations between English and Spanish.
     Your job is to determine whether the translation from English
     to Spanish below has an incorrect literal translation of text
     that uses an English idiom.
     Here are some examples:
     ---
     {few_shot_examples}
     ---
     English: {ENGLISH}
     Spanish: {SPANISH}
     ---
     {format_instructions}
     """,
   input_variables=[ "ENGLISH", "SPANISH"],
   partial_variables={
      "few_shot_examples": examples,
      "format_instructions":parser.get_format_instructions()
   }
)

chain = prompt | model | parser

dataset = "data/idiom_classifier_test.csv"
df = pd.read_csv(dataset)

results = pd.DataFrame()

records = [x for i,x in df.iterrows()] 

for r in tqdm(records, desc=f"Processing Idiom Classifications for {dataset}"):

         record = {"ENGLISH": r['text'], "SPANISH": r['translation']}
         try:
             response = chain.invoke(record)
             r['literal_only'] = int(response['literal_only'])
         except Exception as e:
             print("Error processing record", e)
             r['literal_only'] = np.nan
         results = pd.concat([results, pd.DataFrame([r])], ignore_index=True)

results.to_csv("data/idiom_test_few_shot_model.csv")


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

y_true = list(results['label'])
y_pred = list(results['literal_only'])

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: \t{round(precision,2)}")
print(f"Recall: \t{round(recall,2)}")

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Idiom Positive', 'Idiom Literal'])
disp.plot()
plt.savefig("results/few_shot_cm.png")

