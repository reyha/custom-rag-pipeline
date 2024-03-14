from typing import Any

from PyPDF2 import PdfReader, PdfWriter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


class DataHandler():
    def __init__(self, config) -> None:
        self.config = config

    def extract_relevant_docs(self):
        pdf_file_path = self.config.get("raw_file_path")
        file_base_name = pdf_file_path.replace('.pdf', '')

        pdf = PdfReader(pdf_file_path)
        pdfWriter = PdfWriter()

        # as extracting first two chapter only
        for page_num in range(18, 68):
            pdfWriter.add_page(pdf.pages[page_num])

        with open('{0}_2chapters.pdf'.format(file_base_name), 'wb') as f:
            pdfWriter.write(f)
            f.close()

        documents = SimpleDirectoryReader(
            input_files=["../../Downloads/ConceptsofBiology-WEB_2chapters.pdf"]).load_data()

        return documents

    def get_embed_model(self):
        embed_model_path = self.config.get("embed_model_path")
        embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
        return embed_model

    def get_chunked_docs(self, documents, level, chunk_size):
        if level == "sentence":
            text_parser = SentenceSplitter(
                chunk_size=chunk_size,
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

            # Reduce # nodes to 2 for testing
            nodes = nodes[0:2]

            for node in nodes:
                embed_model = self.get_embed_model()
                node_embedding = embed_model.get_text_embedding(
                    node.get_content(metadata_mode="all")
                )
                node.embedding = node_embedding

            return nodes

    def prepare_data(self):
        documents = self.extract_relevant_docs()
        doc_chunks = self.get_chunked_docs(documents, level="sentence", chunk_size = self.config.get("CHUNK_SIZE", 1024))
        return doc_chunks

    def create_vector_store(self):
        # create client and a new collection
        chroma_client = chromadb.EphemeralClient()
        ch_collection_name = self.config.get("collection", "customragpipeline")
        chroma_collection = chroma_client.get_or_create_collection(ch_collection_name)
        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        data_nodes = self.prepare_data()
        vector_store.add(data_nodes)
        return vector_store






