from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
   template = """
     Your job is to categorise a news article to determine which, if any, 
     of the NAICS industry categories is most relevant to for investors. 
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


from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
chain = prompt | model | parser

lookup = {v: k for k, v in categories.items()}

response = chain.invoke({
   "article":"Last weekend residents of inner city Brisbane Australia were woken by the sound of the XXXX brewery exploding. Beer was sprayed across the city and resulted in the death of local wildlife."
})
code = lookup[response['category']]
print(f"Categorised as {code} : {response['category']} ")

