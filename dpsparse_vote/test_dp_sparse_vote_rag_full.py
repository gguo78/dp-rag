# test_dp_sparse_vote_rag_full.py

from dp_sparse_vote_rag_v2 import DPSparseVoteRAG
from real_retriever import RealRetriever
from real_generator import RealGenerator
from load_dataset import load_triviaqa_subset

# Load TriviaQA subset
questions, answers = load_triviaqa_subset(split='validation', num_samples=50)

# Using answers themselves as "documents"
documents = answers  

# Initialize real retriever and generator
retriever = RealRetriever(documents)
generator = RealGenerator(model_name='facebook/opt-1.3b')

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

# Run the system on a few sample queries
for idx, query in enumerate(questions[:5]):
    print(f"\nQuestion {idx+1}: {query}")
    answer = dp_sparse_rag.generate_answer(query, max_tokens=30)
    print("Generated Answer:", answer)
