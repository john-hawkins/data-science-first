
def get_chain(code, model):
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
   subchain = subprompt | model | subparser
   return subchain


subchain = get_chain(code, model)

response = subchain.invoke({"article":"Last weekend residents of inner city Brisbane Australia were woken by the sound of the XXXX brewery exploding. Beer was sprayed across the city and resulted in the death of local wildlife."})

print(f"Categorised as {response['subcategory']} ")

