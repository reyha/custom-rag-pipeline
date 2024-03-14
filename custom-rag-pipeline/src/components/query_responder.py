from typing import Any

from src.components.base_classes.base_module import BaseModule
from llama_index.core.query_engine import RetrieverQueryEngine


class QueryResponder(BaseModule):
    def __init__(self, service_state: Any) -> None:
        super().__init__(service_state)

    def run_query_responder(self, retriever: Any, llm: Any) -> Any:
        query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
        user_input = self.inputs.get("user_query")
        response = query_engine.query(user_input)
        return response