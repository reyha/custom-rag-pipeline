"""
Document Retriever Module for QnA Service
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.components.data_handler import DataHandler
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.vector_stores import VectorStoreQuery


class BaseRetriever(ABC):
    """ Base Retriever class with methods to be implemented by all Retrievers
    """

    @abstractmethod
    def _retrieve(self, *args, **kwargs) -> str:
        """Get documents relevant for a query.
        Returns:
            Tuple[List, List]: Chosen sections' content and most relevant articles
        """


class VectorDBRetriever(BaseRetriever):
    """ Document Retriever connecting to DocR service
    """

    def __init__(self,
                 service_state,
                 query_mode: str = "default",
                 similarity_top_k: int = 2,
                 ) -> None:
        self.service_state = service_state
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.embed_model = DataHandler(self.service_state['config'])
        self.vector_store = self.service_state['data_store']
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self.embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

