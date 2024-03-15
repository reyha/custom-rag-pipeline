"""
Data handler for QA Pipeline
"""
import chromadb
from PyPDF2 import PdfReader, PdfWriter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class DataHandler():
    def __init__(self, config) -> None:
        self.config = config

    def get_embed_model(self):
        """
        Load embedding model
        """
        embed_model_path = self.config.get("embed_model_path")
        embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
        return embed_model

    def extract_relevant_docs(self):
        """
        Extract text from source i.e. pdf and load it
        in directory reader loader by llamaindex for
        easy access during retrieval.
        """
        pdf_file_path = self.config.get("raw_file_path")
        file_base_name = pdf_file_path.replace('.pdf', '')

        pdf = PdfReader(pdf_file_path)
        pdfWriter = PdfWriter()

        # Extract first two chapter only
        for page_num in range(18, 68):
            pdfWriter.add_page(pdf.pages[page_num])

        with open('{0}_2chapters.pdf'.format(file_base_name), 'wb') as f:
            pdfWriter.write(f)
            f.close()

        documents = SimpleDirectoryReader(
            input_files=['{0}_2chapters.pdf'.format(file_base_name)]).load_data()

        return documents

    def get_chunked_docs(self, documents, level, chunk_size):
        """
        Split source dato into chunks based on level (sentence,
        words, char etc.).
        """

        # 1. Sentence level chunking
        if level == "sentence":
            text_parser = SentenceSplitter(
                chunk_size=chunk_size,
            )
            text_chunks = []
            # maintain relationship with source doc index, to help inject doc metadata
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

            return nodes

    def generate_embeddings(self, nodes):
        """
        Add embedding for each node in docs for ease of retrieval.
        """
        for node in nodes:
            # Initialize embedding model
            embed_model = self.get_embed_model()
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        return nodes

    def prepare_data(self):
        """
        Extract data from source and chunk based on level.
        """
        documents = self.extract_relevant_docs()
        chunk_size = self.config.get("CHUNK_SIZE", 1024)
        doc_chunks = self.get_chunked_docs(documents,
                                           level="sentence",
                                           chunk_size=chunk_size)
        return doc_chunks

    def create_vector_store(self):
        """
        Create vector store with all data nodes and their
        corresponding embeddings.
        """
        # 1. Initialize client
        chroma_client = chromadb.EphemeralClient()
        # 2. Get or create collection
        ch_collection_name = self.config.get("collection", "customragpipeline")
        chroma_collection = chroma_client.get_or_create_collection(ch_collection_name)
        # 3. Set up ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # 4. Prepare data - extract, chunk
        data_chunks = self.prepare_data()
        # 5. Add embedding for chunked data
        # To-Do: Reduced to 5 for testing - remove on prod
        data_chunks = data_chunks[0:5]
        doc_chunks_with_embed = self.generate_embeddings(data_chunks)
        # 6. Add chunked data to vector store
        vector_store.add(doc_chunks_with_embed)
        return vector_store






