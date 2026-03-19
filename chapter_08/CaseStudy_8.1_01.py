import json
import pickle
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class ComplaintTopics(BaseModel):
    topics: list[str] = Field(
        description="A list of complaint topics that are commonly raised by\
                     customers for the given category. Each topic should\
                     provide lots of detail including a comma separated\ 
                     list of potential variations such as hardware types,\
                     account types or regional variations."
    )

parser = JsonOutputParser(pydantic_object=ComplaintTopics)    

prompt = PromptTemplate(
   template = """
     You are an expert on the Telecommunications Industry.
     We will provide a general category of customer complaint.
     We want you to create an exhaustive list of common customer
     complaint topics within that complaint category. 
     You should provide lots of details including a list of any 
     specific equipment or account types that can occur. 
     Use the random seed value {seed} for variation.  
     ---
     Complaint Category: {complaint}
     ---
     {format_instructions}
     """,
   input_variables=["seed", "complaint"],
   partial_variables={
      "format_instructions":parser.get_format_instructions()
   }
)

 
complaints = {
   "account": "Customer Account Issues relating to payments and services",
   "network": "Network Outages including both home and mobile, data and voice services",
   "hardware": "Hardware Failure including handsets, routers and connected devices", 
   "other": "Irrelevant or unfocused complaints that indicate cusomter misunderstanding or unreasonable expectations"
}

from langchain_google_genai import ChatGoogleGenerativeAI
model_name = "gemini-2.5-pro"
model = ChatGoogleGenerativeAI(model=model_name)
chain = prompt | model | parser

for c in complaints.keys():
    topics = []
    for i in range(7):
       response = chain.invoke({"seed":str(i*42), "complaint": complaints[c]})
       data = response['topics']
       topics.extend(data)
    with open(f"data/{c}.txt", "w") as f:
        f.write("\n".join(topics) + "\n") #
    print(f"Generated: {c} \t\t Topics: {len(topics)}")



