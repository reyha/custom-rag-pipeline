"""
Query responder module for QA Pipeline
"""
from typing import Any

from src.components.base_classes.base_module import BaseModule
from llama_index.core.query_engine import RetrieverQueryEngine


class QueryResponder(BaseModule):
    def __init__(self, service_state: Any) -> None:
        super().__init__(service_state)

    def run_query_responder(self, type: str) -> Any:
        """
        With top-n most similar docs from retriever and input query
        from user, generate response using llama-13b model.
        """
        retriever = self.interim_outputs.get("retriever", "")
        llm = self.interim_outputs.get("llm", "")

        # Type "lib" if using libraries such as llamaindex
        if type == "lib":
            query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
            user_input = self.inputs.get("user_query")
            response = query_engine.query(user_input).response
            # Write to Outputs Dict
            self.interim_outputs.update({"response": response})
        # Type "custom" if doing naive implementation [Note: Not yet implemented]
        elif type == "custom":
            """
            If custom, draft an explcit prompt including all possible guards to ensure bot doesn't hallucinate. 
            
            Sample:
            prompt = "You are a search system having an expertise in biology. Your task is to correctly answer
            query mentioned below within quotes. Remember that you can only formulate answer based on the 
            context mentioned below between ###. If your context is empty, you simply say "I am unable to help
            you with this query."" + \n + "context:" + "###" + context + "###" + \n + "query:" + "'''" + query 
            + "'''" 
            """
            pass
        else:
            pass
