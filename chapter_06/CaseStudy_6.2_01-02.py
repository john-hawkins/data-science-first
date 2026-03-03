from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
 
class SelectionCriteria(BaseModel):
    education: str = Field(description="A summary of education and credentials.")
    experience: str = Field(description="A summary of relevant work experience.")
    skills: str = Field(description="A summary of key skills.")

parser = JsonOutputParser(pydantic_object=SelectionCriteria)


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
   template = """
     Your job is to extract selection criteria from a job candidate's resume.
     The complete text of the resume will be provided, you need to identify the
     parts of the resume that contain descriptions of their education, work experience
     and key skills. Each of these three elements should be summarised and returned
     in a separate variables according to the instructions below.
      ---
     Here is the resume for you to process:
     ---
     {resume}.
     ---
     {format_instructions}
     """,
   input_variables=["resume"],
   partial_variables={"format_instructions":parser.get_format_instructions()}
)


def extract_criteria(chain, resume):
   results = {}
   criteria = ["education", "experience", "skills"]
   try:
       response = chain.invoke({"resume":resume})
       for k in criteria: 
           if k in response:
               results[k] = response[k]
           else:
               results[k] = "" 
   except:
       for k in criteria:
           results[k] = "" 
   return results


from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
import pandas as pd

resumes = pd.read_csv("data/resumes.csv")
processed = pd.DataFrame()

chain = prompt | model | parser

for index,row in resumes.iterrows():
    criteria = extract_criteria(chain, row['text'])
    record = {"file":row['file'], 
              "education":criteria["education"], 
              "experience":criteria["experience"],
              "skills":criteria["skills"]
    }
    processed = pd.concat([processed, pd.DataFrame([record])], ignore_index=True)


processed.to_csv("data/resumes_processed.csv", index=False)
