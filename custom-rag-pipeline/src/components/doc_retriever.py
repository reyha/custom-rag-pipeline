"""
Document Retriever Module for QA Pipeline
"""
from abc import ABC
from typing import List

from src.components.data_handler import DataHandler
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.vector_stores import VectorStoreQuery


class BaseRetriever(ABC):
    """ Base Retriever class with methods to be implemented by all Retrievers
    """
    def retrieve(self, *args, **kwargs) -> List[NodeWithScore]:
        """Get docs/nodes relevant for a query.
        Returns:
            List[NodeWithScore]: Document nodes with similarity score
        """


class VectorDBRetriever(BaseRetriever):
    """ Document Retriever for QA Pipeline
    """

    def __init__(self,
                 service_state,
                 query_mode: str = "default",
                 top_n: int = 2,
                 ) -> None:
        self.service_state = service_state
        self._query_mode = query_mode
        self._top_n = top_n
        super().__init__()

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Using embedding model and vector store initialized before,
        retrieve most similar top n docs/nodes

        Args:
            query_bundle: input query

        Return:
            Returns top-n most relevant nodes with similarity scores for each
        """
        # 1. Load embedding model and vector store
        embed_model = DataHandler(self.service_state.config).get_embed_model()
        vector_store = self.service_state.vector_store
        # 2. Get embedding for user query
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        # 3. Retrieve top_n most similar queries
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._top_n,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)
        # 4. Add scores to nodes
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # To-Do: 5. Filter out nodes based on scores i.e. sim_score < threshold,
        # do not include as similar node

        return nodes_with_scores

