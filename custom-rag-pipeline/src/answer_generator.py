import json
from typing import Any
from llama_index.llms.llama_cpp import LlamaCPP

from src.components.base_classes.base_module import BaseModule
from src.components.query_responder import QueryResponder
from src.components.search_handler import SearchHandler


class QAGenerator(BaseModule):
    """Class for Answer Generation for given query powered by OpenAI ChatGPT"""

    def __init__(self, service_state: Any):
        super().__init__(service_state)
        # Placeholder to save intermediate and yield outputs
        self.interim_outputs.update({"answer_id": self.answer_id})
        self.query_responder = QueryResponder(self.service_state)
        self.search_handler = SearchHandler(self.service_state)
        self.to_yield = {}

    def prepare(self):
        try:
            # 3. Prepare context - get docs from vector store
            retriever = self.search_handler.get_doc_retriever()
            llm = LlamaCPP(
                model_path=self.config['llm_model_path'],
                temperature=0.1,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                verbose=True,
            )
            self.interim_outputs.update({"retriever": retriever})
            self.interim_outputs.update({"llm": llm})
        except Exception as error:
            self.interim_outputs.update({"error": str(error)})
            raise

    def generate(self):
        retriever = self.interim_outputs.get("retriever", "")
        llm = self.interim_outputs.get("llm", "")

        response = self.query_responder.run_query_responder(retriever, llm)

        user_query = self.inputs.get("user_query", "")
        json_string = json.dumps(
            {
                "response": response,
                "user_query": user_query,
                "answer_id": self.answer_id
            },
            separators=(",", ":"),
        )
        return json_string