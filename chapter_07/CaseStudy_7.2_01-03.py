### Chapter 7 - Case Study 7.2 - LISTING 5

import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

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

