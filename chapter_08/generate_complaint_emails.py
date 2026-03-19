import re
import json
import pickle
import random
from tqdm import tqdm
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
 

class ComplaintEmail(BaseModel):
    email: str = Field(
        description="A generated complaint email from a customer that\
                     fits the provided description.\
                     Ensure that the writing style reflects the\
                     properties of the customer and\
                     the subject of the complaints fits the description\
                     of the issues described."
    )

parser = JsonOutputParser(pydantic_object=ComplaintEmail)

prompt = PromptTemplate(
   template = """
     You should play the role of a customer of the {COMPANY}
     telecommunications company writing a complaint email.
     You are a {GENDER} aged {AGE} with a highest education 
     level of {EDU}.
     You have been a customer for {COMPANY} for {TENURE} years. 
     You are writing them to complain about the general topic 
     of {CATEGORY} issues. 
     The specific issue you are experiencing is described below
     along with some specific contributing detail. 
     Write the complaint email to the company with a sense of 
    {EMOTION} in your writing.
     ---
     Issue: {ISSUE}
     Detail: {DETAIL}
     ---
     {format_instructions}
     """,
   input_variables=[ 
      "COMPANY", "GENDER", "AGE", "EDU", "TENURE", 
      "CATEGORY", "ISSUE", "DETAIL", "EMOTION"
   ],
   partial_variables={
      "format_instructions":parser.get_format_instructions()
   }
)


class JsonCleaner(BaseOutputParser):
    _field_name = None
    def __init__ (self, field_name, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._field_name = field_name

    def parse(self, text: str) -> str:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_string = match.group(0)
            return json_string
        else:
            return "{ \"" + self._field_name + "\": \"" + text + "\" }"


jsonclean = JsonCleaner("email")


class COTCleaner(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(r'</think>', text, re.DOTALL)
        if match:
            return text[match.span()[1]:]
        else:
            return text


cotclean = COTCleaner()

from langchain_ollama import ChatOllama
model = ChatOllama(model="deepseek-r1:8b", reasoning=False)

chain = prompt | model | cotclean | jsonclean | parser


def get_list_from_file(file):
    filename = f"data/{file}"
    with open(filename, "r") as f:
        my_list = [line.strip() for line in f.readlines()]
    return my_list


def random_list_item(input):
   temp = random.randint(0, len(input)-1)
   return input[temp]


samples = 3
companies = get_list_from_file("companies.txt")
genders = get_list_from_file("genders.txt")
educations = get_list_from_file("educations.txt")
emotions = get_list_from_file("emotions.txt")

#complaints = ["account", "network", "hardware", "other"]
complaints = [ "account"]
#, "other", "network", "account"]

results = pd.DataFrame()

for c in complaints:
    filename = f"{c}.txt"
    topic_list = get_list_from_file(filename)
 
    for t in tqdm(topic_list, desc=f"Generating for {filename}"):
        issue = t
        detail = ""
        if "Variations:".casefold() in t.casefold():
           temp = t.split("Variations:".casefold())
           issue = temp[0]
           if len(temp)> 1:
              variants = temp[1].split(",")
              vi = random.randint(0,len(variants)-1)
              detail = variants[vi]
        for i in range(0, samples):
            company = random_list_item(companies)
            gender = random_list_item(genders)
            age = str(random.randint(18,80))
            edu = random_list_item(educations)
            tenure = str(random.randint(1,10))
            emotion = random_list_item(emotions)
            record =                 {
                "COMPANY": company,
                "GENDER": gender,
                "AGE": age,
                "EDU":  edu,
                "TENURE": tenure,
                "CATEGORY": c,
                "ISSUE": issue,
                "DETAIL": detail,
                "EMOTION": emotion
            }

            try:
                response = chain.invoke(record)
                if 'email' in response: 
                    record['email'] = response['email']
                else:
                    record['email'] = "INVALID"
            except Exception as e:
                record['email'] = "ERROR"
                print("ERROR IN MODEL INVOCATION")
                print(e)
            results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)
    
results.to_csv("data/generated_complaint_emails_account.csv", index=False)

