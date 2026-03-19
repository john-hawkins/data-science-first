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

