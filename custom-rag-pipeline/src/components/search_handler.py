from typing import Any

from src.utils.exceptions import CustomException
from src.components.base_classes.base_module import BaseModule
from src.components.doc_retriever import VectorDBRetriever


class SearchHandler(BaseModule):
    def __init__(self, service_state: Any) -> None:
        super().__init__(service_state)

    def get_doc_retriever(self):
        doc_retriever = self.config.get("DOC_RETRIEVER", "")
        if doc_retriever == "doc_retrieval_ada":
            top_n = self.config.get("TOP_N")
            retriever = VectorDBRetriever(self.service_state, similarity_top_k=top_n)
            return retriever
        raise CustomException(
            500,
            "Issue in initialising doc_retriever: "
            f"DOC_RETRIEVER value {doc_retriever} in config is invalid"
        )
