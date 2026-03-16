import json
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
#from langchain.output_parsers.enum import EnumOutputParser

with open("data/2022_NAICS_Roots.json", 'r') as f:
    categories = json.load(f) 

categories["00"] = "Unknown industry or content not related to any industry"

CategoryEnum = Enum('CategoryEnum', categories) 

# Define a Pydantic model using the dynamically created Enum
class NAICS_Category(BaseModel):
    category: CategoryEnum = Field(description="The NAICS industry category.")
    reason: str = Field(description="An explanation for the assigned category.")

parser = JsonOutputParser(pydantic_object=NAICS_Category)

