"""
Answer Generation module powered by Llama-13b
"""
import json
import time
from typing import Any

from src.components.base_classes.base_module import BaseModule
from src.components.query_responder import QueryResponder
from src.components.search_handler import SearchHandler
from src.components.llm_handler import LLMHandler
from src.utils.exceptions import CustomException


class QAGenerator(BaseModule):
    """Class for Answer Generation for given query powered by Llama-13b"""

    def __init__(self, service_state: Any):
        super().__init__(service_state)
        # Placeholder to save intermediate and yield outputs
        self.interim_outputs.update({"answer_id": self.answer_id})
        self.query_responder = QueryResponder(self.service_state)
        self.search_handler = SearchHandler(self.service_state)
        self.llm_handler = LLMHandler(self.service_state)

    def package_response(self):
        user_query = self.inputs.get("user_query", "")
        response = self.interim_outputs.get("response", "")
        json_string = json.dumps(
            {
                "response": response,
                "user_query": user_query,
                "answer_id": self.answer_id
            },
            separators=(",", ":"),
        )
        return json_string

    def prepare(self):
        """
        Prepare for response generation by initializing
        retriever and LLM i.e. llama-13b.
        """
        try:
            # 1. Initialize retriever and save to output dict
            retriever = self.search_handler.get_doc_retriever()
            # 2. Initialize LLM and save to output dict
            llm = self.llm_handler.get_llm()
            # 3. Write to Output Dict
            self.interim_outputs.update({"retriever": retriever})
            self.interim_outputs.update({"llm": llm})
        except Exception as error:
            self.interim_outputs.update({"error": str(error)})
            raise

    def generate(self):
        """
        Generate response and package as json.
        """
        # 1. Run query responder
        # a. If type set to "lib", pipeline will use llamaindex
        # b. [To-Do] If type set to "custom", pipeline will load prompts from prompt factory and call llm
        try:
            self.query_responder.run_query_responder(type="lib")
        except Exception:
            self.logger.error(
                f"Error - InternalError - answer_id:{self.answer_id}:msg:issue_getting_llm_response"
            )
            raise CustomException(
                500,
                "Issue in getting response from llm"
            )

        # 2. Package response as json
        output = self.package_response()
        self.logger.info(
            f"answer_id:{self.answer_id}: time taken to generate result : "
            f"{(time.time() - self.app_start_time)} : "
        )
        return output