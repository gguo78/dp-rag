# real_retriever.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RealRetriever:
    def __init__(self, documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.index = self.build_faiss_index(documents)

    def build_faiss_index(self, documents):
        # Encode all documents into dense vectors
        doc_embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        dimension = doc_embeddings.shape[1]
        # Build FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)
        self.doc_embeddings = doc_embeddings  # Save for later use if needed
        return index

    def retrieve(self, query, k=5):
        # Encode query into dense vector
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # Search for top-k similar documents
        distances, indices = self.index.search(query_embedding, k)
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs