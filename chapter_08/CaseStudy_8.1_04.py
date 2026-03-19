import re
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser


class JsonCleaner(BaseOutputParser):
    _field_name = None
    def __init__ (self, field_name, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._field_name = field_name

    def parse(self, text: str) -> str:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_string = match.group(0)
            return json_string
        else:
            return "{ \"" + self._field_name + "\": \"" + text + "\" }"


jsonclean = JsonCleaner("email")


class COTCleaner(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(r'</think>', text, re.DOTALL)
        if match:
            return text[match.span()[1]:]
        else:
            return text


cotclean = COTCleaner()

from langchain_ollama import ChatOllama
model = ChatOllama(model="deepseek-r1:8b", reasoning=False)

chain = prompt | model | cotclean | jsonclean | parser

