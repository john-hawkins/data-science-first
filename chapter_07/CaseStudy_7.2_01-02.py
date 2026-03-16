### Chapter 7 Case Study 7.2 - LISTING 4

from typing import Callable, Dict
from langchain_core.runnables import RunnableSequence


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
   ("vocabulary", vocab_score)
]

for essay in test_essays:
    response = custom_invoke(chain, "essay", essay, procs)
    print(response)
    print("\n")

