import json
import functools
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Dict

class RetrieveCode(Runnable):
    def __init__(self, lookup, in_field, out_field, default="?"):
        self.lookup = lookup
        self.in_field = in_field
        self.out_field = out_field
        self.default = default

    def invoke(
        self,
        input: Any,
        config: RunnableConfig = None,
        **kwargs: Any,
    ) -> Any:
        # Implement the custom logic here
        if input[self.in_field] in self.lookup:
            input[self.out_field] = self.lookup[ input[self.in_field] ]
        else:
            input[self.out_field] = self.default
        return input


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


@functools.lru_cache(maxsize=None)
def get_root_chain():
   with open("data/2022_NAICS_Roots.json", 'r') as f:
       categories = json.load(f)
   categories["00"] = "Unknown industry or content not related to any industry"
   CategoryEnum = Enum('CategoryEnum', categories)
   class NAICS_Category(BaseModel):
       category: CategoryEnum = Field(description="The NAICS industry category.")
       reason: str = Field(description="An explanation for the assigned category.")

   parser = JsonOutputParser(pydantic_object=NAICS_Category)
   prompt = PromptTemplate(
      template = """
        Your job is to categorise a news article to determine which, if any,
        of the NAICS industry categories it is most relevant to for investors.
        You are not categorising the content itself, but whether the news
        story contains information that will be pertinent to investors in
        a specific sector of the economy. Return the name of the industry
        category from the provided list of options.

        Here is data for you to process:
        ---
        article: {article}
        ---
        {format_instructions}
        """,
      input_variables=["article"],
      partial_variables={"format_instructions":parser.get_format_instructions()}
   )
   lookup = {v: k for k, v in categories.items()}
   post_process = RetrieveCode(lookup, "category", "code")
   chain = prompt | model | parser | post_process
   return chain


@functools.lru_cache(maxsize=None)
def get_chain(code):
   file_name = "data/2022_NAICS_"+code+".json"
   with open(file_name, 'r') as f:
      subcats = json.load(f)

   SubCatEnum = Enum('SubCategoryEnum', subcats)
   class SubCategory(BaseModel):
      subcategory: SubCatEnum = Field(description="The NAICS industry sub category.")
      reason: str = Field(description="An explanation for the assigned category.")

   subparser = JsonOutputParser(pydantic_object=SubCategory)

   subprompt = PromptTemplate(
      template = """
        Your job is to identify the most appropriate sub-category from the
        NAICS industry categories for the news article.
        You are not categorising the content itself, but whether the news
        story contains information that will be pertinent to investors in
        a specific sector of the economy. Return the name of the industry
        sub-category from the provided list of options.

        Here is data for you to process:
        ---
        article: {article}
        ---
        {format_instructions}
        """,
      input_variables=["article"],
      partial_variables={"format_instructions":subparser.get_format_instructions()}
   )

   sublookup = {v: k for k, v in subcats.items()}
   post_process = RetrieveCode(sublookup, "subcategory", "code")
   subchain = subprompt | model | subparser | post_process
   return subchain


def categorize_text(input_text, max_depth=2):
    chain = get_root_chain()
    response = chain.invoke({"article":input_text})
    code = response['code']
    code_len = len(code)
    result = {}
    name = "lvl"+str(code_len)
    result[name] = code
    if code != "00":
       while code_len <= max_depth:
          subchain = get_chain(code)
          response2 = subchain.invoke({"article":input_text})
          code = response2['code']
          code_len = len(code)
          name = "lvl"+str(code_len)
          result[name] = code
    return result

