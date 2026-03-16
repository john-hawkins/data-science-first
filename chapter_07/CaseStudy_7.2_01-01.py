### Chapter 7 Case Study 7.2 -  LISTING 1

import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Essay_Features(BaseModel):
    embedded_clause: int = Field(
        description="Count of the number of embedded clauses."
    )
    passive_voice: int = Field(
        description="Count of the instances of passive voice."
    )
    long_sentences: int = Field(
        description="Count of sentences longer than 20 words."
    )
    obscure_vocab: int = Field(
        description="Count of obscure vocabulary used."
    )
    reasoning_lines: int = Field(
        description="Count of the number of lines of reasoning used."
    )
    unusual_grammar: int = Field(
        description="Count of unusual grammatical constructs."
    )
    has_adjectives: int = Field(
        description="Indicator variable for presence of adjectives"
    )
    has_adverbs: int = Field(
        description="Indicator variable for presence of adverbs."
    )
    has_synonyms: int = Field(
        description="Indicator variable for presence of synonyms."
    )
    repeated_words: int  = Field(
        description="Count of the number of times key semantic words are\
                     repeated (not including stop words)"
    )
    detailed_vocab: int  = Field(
        description="Count of the number of detailed words used, they\
                     should be well known but very specific words e.g.\
                     ecstatic, depressed, or resentful to describe a mood."
    )
    rare_vocab: int  = Field(
        description="Indicator variable to show that the text contains\
                     rare words, these are detailed words that are\
                     uncommon in everday usage. e.g.\
                     elated, solemn or vindictive to describe a mood."
    )
    literary_vocab: int = Field(
        description="Indicator variable to show taht the text contains\
                     some literary words, these should be words that are\
                     typically only used in literary\
                     writing e.g. chagrin, enuui or vicissitudes."
    )
    turn_of_phrase: int = Field(
        description="Indicator variable to show in the text contains\
                     unique (non-cliche) turns of phrase. These should\
                     be a short expressive phrases used in the text."
    )
    disconnections: int = Field(
        description="Count of the number of times a sentence does not flow\
                    logically from the preceding text block."
    )
    contradictions: int = Field(
        description="Count of the number of contradictions in the text."
    )
    tense_discontinuity: int = Field(
        description="Count of number of times that the text changes tense,\
        via verb conjugation."
    )


parser = JsonOutputParser(pydantic_object=Essay_Features)


### Chapter 7 Case Study 7.2 - LISTING 2

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
   template = """
     You are an English language expert, your job is to process a student
     essay and calculate multiple numerical features that evaluate the 
     quality of their writing. 
     These features involve looking at the grammatical elements of the 
     text, the choice of words used and the logical structure of the 
     argument or story that is presented.

     Here is data for you to process:
     ---
     Essay: {essay}
     ---
     {format_instructions}
     """,
   input_variables=["essay"],
   partial_variables={
      "format_instructions":parser.get_format_instructions()
   }
)


### Chapter 7 Case Study 7.2 -  LISTING 3

from langchain_google_genai import ChatGoogleGenerativeAI
model_name = "gemini-2.5-pro"
model = ChatGoogleGenerativeAI(model=model_name)
chain = prompt | model | parser

test_essays = [
""" 
Arriving in a new town, which can be both exciting and daunting, brings with it a sense of adventure. As I stepped off the bus, which had been delayed for hours, I immediately noticed the bustling streets filled with unfamiliar faces. Finding my way to the hotel, where I would be staying for the next few days, proved to be more challenging than I could possibly have anticipated. Despite feeling a bit bamboozled, I couldn't help but marvel at the phenomenal architecture that adorned the buildings lining the cobblestone streets. Eventually, I stumbled upon a quaint Parisian cafe, where I decided to take a moment to sip tea, soak in the atmosphere and plan my next steps.
""",
"""
Arriving in a new town brings with it a sense of adventure. I stepped off the bus and saw streets filled with people. Finding my way to the hotel was hard. I felt a bit lost, but still managed to admire the buildings. I found a small cafe and planned my next steps.
"""
]

for essay in test_essays: 
    response = chain.invoke({"essay": essay})
    print(response)
    print("\n")

