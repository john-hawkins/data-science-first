
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

