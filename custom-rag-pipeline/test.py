from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from PyPDF2 import PdfReader, PdfWriter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.vector_stores import VectorStoreQuery

import streamlit as st
st.title("Biology Guide")
user_input = st.text_input("Enter your question about biology:", "")

# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_path="../../Downloads/llama-2-13b-chat.Q4_0.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
#     model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
#     model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

embed_model = HuggingFaceEmbedding(model_name="../../Downloads/all-mpnet-base-v2")


pdf_file_path = '../../Downloads/ConceptsofBiology-WEB.pdf'
file_base_name = pdf_file_path.replace('.pdf', '')

pdf = PdfReader(pdf_file_path)
pdfWriter = PdfWriter()

for page_num in range(18, 68):
    pdfWriter.add_page(pdf.pages[page_num])

with open('{0}_2chapters.pdf'.format(file_base_name), 'wb') as f:
    pdfWriter.write(f)
    f.close()

documents = SimpleDirectoryReader(input_files=["../../Downloads/ConceptsofBiology-WEB_2chapters.pdf"]).load_data()

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)
    
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding
    
# create client and a new collection
chroma_client = chromadb.EphemeralClient()

chroma_collection = chroma_client.get_or_create_collection("new_coll_v2")

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

vector_store.add(nodes)

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
    
retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

if st.button("Submit"):
    try:
        response = query_engine.query(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")