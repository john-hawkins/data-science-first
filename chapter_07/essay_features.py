from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Callable, Dict
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os

import claude_feature_funcs as claude

model_name = "gemini-2.5-pro"

model = ChatGoogleGenerativeAI(model=model_name)

class Essay_Features(BaseModel):
    embedded_clause: int = Field(
        description="Count of the number of embedded clauses."
    )
    passive_voice_use: list[str] = Field(
        description="Extracted instances of verbs in passive voice from the text.\
                     These should be any instances of a verb in which a verb phrase is passive."
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
    obscure_vocab_use: list[str] = Field(
        description="Extracted words from the text that are considered obscure vocabulary."
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
    repeated_words_use: list[str] = Field(
        description="Extracted instances of repeated words in the text.\
                     These should be meaningful and significant words not including\
                     stop words such as prepositions and pronouns."
    )
    detailed_vocab: int  = Field(
        description="Count of the number of detailed words used, they\
                     should be well known but very specific words e.g.\
                     ecstatic, depressed, or resentful to describe a mood."
    )
    detailed_vocab: int  = Field(
        description="Count of the number of detailed words used, they\
                     should be well known but very specific words e.g.\
                     ecstatic, depressed, or resentful to describe a mood."
    )
    detailed_vocab_used: list[str]  = Field(
        description="Extracted instances of detailed vocabulary words used in the text.\
                     These should be widely known but very specific words e.g.\
                     ecstatic, depressed, or resentful to describe a mood."
    )
    rare_vocab: int  = Field(
        description="Indicator variable to show that the text contains rare words, these are\
                     detailed words that are uncommon in everday usage e.g.\
                     elated, solemn or vindictive to describe a mood."
    )
    rare_vocab_used: list[str]  = Field(
        description="Extracted instances of rare vocabulary words used in the text.\
                     These words should be very specific in meaning and uncommon in everday usage e.g.\
                     elated, solemn or vindictive to describe a mood."
    )
    literary_vocab: int = Field(
        description="Indicator variable to show taht the text contains some literary words, these\
                     should be words that typically only used in literary\
                     writing e.g. chagrin, enuui or vicissitudes."
    )
    literary_vocab_used: list[str] = Field(
        description="Extracted instances of literary words used in the text.\
                     These should be words that typically only used in literary\
                     writing e.g. chagrin, enuui or vicissitudes."
    )
    turn_of_phrase: int = Field(
        description="Indicator variable to show in the text contains unique (non-cliche) turns of\
                     phrase. These should be a short expressive phrases used in the text."
    )
    turn_of_phrase_used: list[str] = Field(
        description="Extracted instances unique (non-cliche) turns of\
                     phrase used in the text. Can be quotations as long as they are not common."
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

chain = prompt | model | parser

def custom_invoke (
  chain:RunnableSequence,
  var: str, text:str,
  funcs: list[(str,Callable)]
 ):
    response = chain.invoke({var:text})
    for f in funcs:
        response[f[0]] = f[1](text, response)
    return response

def sentence_count(x:str, r:Dict):
    return len(x.split("."))

def word_count(x:str, r:Dict):
    return len(x.split(" "))

def readbility_score(x:str, r:Dict):
    probs = 2*(r['embedded_clause']/r['sentence_count'])+\
            (r['passive_voice']/r['sentence_count'])+\
            2*(r['long_sentences']/r['sentence_count'])+\
            (r['obscure_vocab']/(r['word_count']/2))+\
            2*(r['reasoning_lines']/r['sentence_count'])+\
            (r['unusual_grammar']/r['sentence_count'])
    if probs>10:
        return 0
    else:
        return 10 - probs

def vocab_score(x:str, r:Dict):
    base_vocab = r['has_adjectives']+r['has_adverbs']+r['has_synonyms']
    detail_vocab = 4*(r['detailed_vocab']/(r['word_count']/2))
    bonus_vocab= r['rare_vocab']+r['literary_vocab']+r['turn_of_phrase']
    repeats = int(r['repeated_words']/(r['word_count']/2))
    score = base_vocab + detail_vocab + bonus_vocab - repeats
    return score

procs = [
   ("word_count", word_count),
   ("sentence_count", sentence_count),
   ("readability", readbility_score),
   ("vocabulary", vocab_score),
   ("flesch_reading_ease", claude.flesch_reading_ease),
   ("flesch_kincaid_grade_level", claude.flesch_kincaid_grade_level),
   ("automated_readability_index", claude.automated_readability_index),
   ("gunning_fog_index", claude.gunning_fog_index),
   ("smog_readability", claude.smog_readability),
   ("type_token_ratio", claude.type_token_ratio),
   ("moving_average_ttr", claude.moving_average_ttr),
   ("yule_k_characteristic", claude.yule_k_characteristic),
   ("average_word_length", claude.average_word_length),
   ("syllable_complexity", claude.syllable_complexity),
   ("logical_connector_density", claude.logical_connector_density),
   ("argument_structure_score", claude.argument_structure_score),
   ("coherence_score", claude.coherence_score),
   ("contradiction_detection_score", claude.contradiction_detection_score)
] 


def process_text(text):
     response = custom_invoke(chain, "essay", text, procs)
     return response

def main(input_path, output_path):
    results = pd.DataFrame()
    df = pd.read_csv(input_path)
    
    essays = [x for i,x in df.iterrows()]
    for row in tqdm(essays, desc=f"Processing file in: {input_path}"):
        try:
           record = process_text(row['text'])
           record["source"] = row['source'] 
           record["generated"] = row['generated']
           record["RANDOM"]= row['RANDOM']
           record["text"]= row['text']
           results = pd.concat([results,pd.DataFrame([record])], ignore_index=True)
        except Exception as e:
           print(e)
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataframe of essays into features.")
    parser.add_argument('in_dir', type=str, help='Path to data.')
    parser.add_argument('out_file', type=str, help='The output file.')
    args = parser.parse_args()
    main(args.in_dir, args.out_file)

