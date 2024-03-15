"""
Search handler for QA Pipeline
"""
from typing import Any

from src.utils.exceptions import CustomException
from src.components.base_classes.base_module import BaseModule
from src.components.doc_retriever import VectorDBRetriever


class SearchHandler(BaseModule):
    def __init__(self, service_state: Any) -> None:
        super().__init__(service_state)

    def get_doc_retriever(self):
        retriever = self.config.get("RETRIEVER", "")
        if retriever == "doc_retriever":
            retriever = VectorDBRetriever(self.service_state,
                                          top_n=self.config.get("TOP_N"))
            return retriever
        raise CustomException(
            500,
            "Issue in initialising doc_retriever: "
            f"DOC_RETRIEVER value {retriever} in config is invalid"
        )
