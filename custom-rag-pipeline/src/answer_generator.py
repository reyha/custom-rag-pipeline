"""
Answer Generation module powered by Llama-13b
"""
import json
import time
from typing import Any

from llama_index.llms.llama_cpp import LlamaCPP

from src.components.base_classes.base_module import BaseModule
from src.components.query_responder import QueryResponder
from src.components.search_handler import SearchHandler


class QAGenerator(BaseModule):
    """Class for Answer Generation for given query powered by Llama-13b"""

    def __init__(self, service_state: Any):
        super().__init__(service_state)
        # Placeholder to save intermediate and yield outputs
        self.interim_outputs.update({"answer_id": self.answer_id})
        self.query_responder = QueryResponder(self.service_state)
        self.search_handler = SearchHandler(self.service_state)
        self.to_yield = {}

    def prepare(self):
        """
        Prepare for response generation by initializing
        retriever and LLM i.e. llama-13b.
        """
        try:
            # 1. Initialize retriever
            retriever = self.search_handler.get_doc_retriever()
            # 2. Initialize LLM
            llm = LlamaCPP(
                model_path=self.config['llm_model_path'],
                temperature=1.0,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                verbose=True,
            )
            # 3. Write to Outputs Dict
            self.interim_outputs.update({"retriever": retriever})
            self.interim_outputs.update({"llm": llm})
        except Exception as error:
            self.interim_outputs.update({"error": str(error)})
            raise

    def generate(self):
        """
        Generate response and package as json.
        """
        # 1. Read from Outputs Dict
        retriever = self.interim_outputs.get("retriever", "")
        llm = self.interim_outputs.get("llm", "")

        # 2. Run query responder
        # Note:
        # 1. If type set to "lib", pipeline will use llamaindex
        # 2. [To-Do] If type set to "custom", pipeline will load prompts from prompt factory and call llm
        response = self.query_responder.run_query_responder("lib", retriever, llm)

        # 3. Package output as json
        user_query = self.inputs.get("user_query", "")
        self.logger.info(
            f"answer_id:{self.answer_id}: time taken to generate result : "
            f"{(time.time() - self.app_start_time)} : "
        )
        json_string = json.dumps(
            {
                "response": response,
                "user_query": user_query,
                "answer_id": self.answer_id
            },
            separators=(",", ":"),
        )
        return json_string