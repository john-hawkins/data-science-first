from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
 
class CriteriaMatch(BaseModel):
    result: bool = Field(description="Binary indicator for selection criteria match")
    score: int = Field(description="A numeric value between 0 and 100 indicating the strength of the match.")
    reason: str = Field(description="A text explanation for the score provided.")

parser = JsonOutputParser(pydantic_object=CriteriaMatch)


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
   template = """
     Your job is to evaluate whether a job candidate matches a particular selection criteria
     based on an extract from their resume. We will provide a description of the selection criteria
     and the relevant excerpt from their resume. You should return a boolean variable to indicate
     if they match, and a score between 0 and 100 baed on the strength of the match. In addition,
     provide a reason that explains the score that was given.
     Here is an example:
     ---
     criteria: Education : A degree in either Computer Science or Engineering.
     resume: I am studying bachelors degree in computer science and I will complete it in 2027.
     result: False
     score: 40
     reason: This candidate is studying the required degree, but they have not yet completed it.  
     ---
     criteria: Experience: Candidate should have at least three years experience building iOS apps. 
     resume: Exmployer ACME corp: iOS developer 2020-2022, Android developer 2022-2025. 
     result: True
     score: 80
     reason: The candidate has done approximately 3 years of iOS development, but it is not recent experience. 
     --------
     Here is data for you to process:
     ---
     criteria: {criteria}
     resume: {resume}
     ---
     {format_instructions}
     """,
   input_variables=["criteria", "resume"],
   partial_variables={"format_instructions":parser.get_format_instructions()}
)



def evaluate_criteria(chain, criteria, resume):
   results = {}
   fields = ["result", "score", "reason"]
   try:
       response = chain.invoke({"criteria":criteria, "resume":resume})
       for k in fields: 
           if k in response:
               results[k] = response[k]
           else:
               results[k] = "" 
   except:
       for k in fields:
           results[k] = "" 
   return results


from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import json

with open("job_listing.json", 'r') as f:
    criteria = json.load(f)

resumes = pd.read_csv("data/resumes_processed.csv")
scored = pd.DataFrame()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
chain = prompt | model | parser

for index,row in resumes.iterrows():
    record = {"file":row['file']}
    for k in criteria.keys():
        crit = criteria[k]
        match = evaluate_criteria(chain, crit, row[k])
        prefix = k[0:3]
        for r in match.keys():
            record[prefix+"_"+r] = match[r]
    scored = pd.concat([scored, pd.DataFrame([record])], ignore_index=True)


scored.to_csv("data/resumes_scored.csv", index=False)

scored['total'] = scored['edu_score'] + scored['ski_score'] + scored['exp_score'] 


import matplotlib.pyplot as plt

plt.hist(scored['total']) 
plt.xlabel("Total Score")
plt.ylabel("Number of Candidates")
plt.title("Distribution of Selection Criteria Scores")
plt.show()


max_score =  scored['total'].max()
best_candidate = scored.index[ scored['total'] == max_score]
best = scored.loc[best_candidate.values[0],:]
 
for col in ['edu_reason', 'ski_reason', 'exp_reason']:
    best[col] = best[col][:55]

print(best.to_markdown())

