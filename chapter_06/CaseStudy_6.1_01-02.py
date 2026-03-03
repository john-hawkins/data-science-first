from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    sentiment: str = Field(description="Categorisation of the sentiment of text as one of these three values: positive, neutral, negative")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

par = JsonOutputParser(pydantic_object=Sentiment)

prompt = PromptTemplate(
   template = """
     Your job is to classify text from a website comment section based on 
     the sentiment expressed by the author. The allowed responses are:
      'positive' - for text expressing positive emotions.
      'negative' - for text expressing negative emotions.
      'neutral' -  for text that expresses no particular emotional content.
     Here are some examples:
      ---
      Text: 'Ohh man, I cannot get enough of this!'
      Response: 'positive'
      Explanation: The author is implying that the content is so good they want more.                          
      ---
      Text: 'This article sucks, the author is stupid.' 
      Response: 'negative'
      Explanation: Sucks is a colloquial expression of negative emotion.
      ---
      Text: 'Shut up and take my money!'
      Response: 'positive'
      Explanation: The phrase shut up is used playfully to express a strong positive response. 
      ---
      Text: 'Where did they find this guy?'
      Response: 'negative'
      Explanation: This question implies that the person is unusually bad.                 
      ---
      Text: 'Can someone tell me what the second part means?'
      Response: 'neutral'
      Explanation: This is a simple question about the content with no emotional loading.                          
      ---
     Here is the text for you to classify:
     {input_text}.
     {format_instructions}
     """,
   input_variables=["input_text"],
   partial_variables={"format_instructions":par.get_format_instructions()}
)


prompt = PromptTemplate(
   template = """
     Your job is to classify text from a website comment section based on
     the sentiment expressed by the author. The allowed responses are:
      'positive' - for text expressing positive emotions.
      'negative' - for text expressing negative emotions.
      'neutral' -  for text that expresses no particular emotional content.
     Here are some examples:
      ---
      Text: 'Ohh man, I cannot get enough of this!'
      Response: 'positive'
      ---
      Text: 'This article sucks, the author is stupid.' 
      Response: 'negative'
      ---
      Text: 'Shut up and take my money!' 
      Response: 'positive'
      ---
      Text: 'Where did they find this guy?'
      Response: 'negative'
      ---
      Text: 'Can someone tell me what the second part means?'
      Response: 'neutral'
      ---
     Here is the text for you to classify:
     {input_text}.
     {format_instructions}
     """,
   input_variables=["input_text"],
   partial_variables={"format_instructions":par.get_format_instructions()}
)

input = "some text to classify "
print(prompt.invoke({"input_text":input}))



from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)

chain = prompt | model | pars

response1 = chain.invoke({"input_text":"Yeah, well, this was a pretty dull read."})
response2 = chain.invoke({"input_text":"OMG, where did she get all these ideas?"})

print("Response 1:", response1['sentiment'], "\nResponse 2:", response2['sentiment'])

import pandas as pd

allowed_values = ["positive", "neutral", "negative"]
def test_model(model, prompt, parser, in_ds, in_col, comp_col, out_col ):
   result = pd.DataFrame(columns=[in_col, comp_col, out_col])
   chain = prompt | model | parser
   for x in range(len(in_ds)):
      text = in_ds[x][in_col]
      comp = in_ds[x][comp_col]
      output = ""
      try:
          response = chain.invoke({"input_text":text})
          if out_col in response:
              output = response[out_col]
              if output not in allowed_values:
                  output = "invalid"
          else:
              output = "missing"
      except:
          output = "error"
      record = {in_col:text, comp_col:comp, out_col:output}
      result = pd.concat([result,pd.DataFrame([record])], axis=0, ignore_index=True)
   return result


test_results = test_model(model, prompt, par, case_ds, "text", "proxy", "sentiment" )


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

oai = ChatOpenAI(model="gpt-4o", temperature=0)
gem = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

models = {
  "gemini-1.5-pro": gem,
  "gpt-4o": oai
}

test_results = pd.DataFrame()

for m in models.keys():
    mod = models[m]
    rez = test_model(
        mod, prompt, par, case_ds, 
        "text", "proxy", "sentiment"
    )
    rez['model'] = m
    test_results = pd.concat(
        [test_results, rez], axis=0, 
        ignore_index=True
    )
 
temp = test_results[ (test_results ['sentiment']=='positive') &  
                     (test_results ['proxy']==1.0)
]

uniqs = temp['text'].unique()
print("Records", len(uniqs), "\n---")
print("\n---\n".join(uniqs))

