# test_dp_sparse_vote_rag.py

from dp_sparse_vote_rag import DPSparseVoteRAG
from toy_models import ToyRetriever, ToyGenerator

# Create synthetic documents
documents = [
    "The cat sat on the mat.",
    "The dog barked at the mailman.",
    "Insurance claims must be filed within 30 days.",
    "Financial documents must remain confidential.",
    "The quick brown fox jumps over the lazy dog."
]

# Initialize retriever and generator
retriever = ToyRetriever(documents)
generator = ToyGenerator()

# Initialize DPSparseVoteRAG engine
dp_sparse_rag = DPSparseVoteRAG(
    retriever=retriever, 
    generator=generator, 
    num_voters=5, 
    k=1,
    epsilon_total=10,
    delta_total=1e-5,
    threshold=0.5
)

# Run the system
query = "What is the story about?"
answer = dp_sparse_rag.generate_answer(query)

print("Generated Answer:", answer)
