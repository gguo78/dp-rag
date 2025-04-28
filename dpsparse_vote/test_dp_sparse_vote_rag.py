# test_dp_sparse_vote_rag.py

import json
import os
from dpsparse_vote.dp_sparse_vote_rag_engine import DPSparseVoteRAGEngine
from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from model import Model
from data import Dataset

# Parameters
num_questions = 50
k = 5
retrieval_epsilon = 1.0
generation_epsilon = 10.0
max_tokens = 30
threshold = 0.5

# Load dataset
dataset = Dataset(num_patients=50, num_diseases=10)  # adjust numbers as needed
rows = list(dataset.rows())[:num_questions]

questions = [row["patient_request"] for row in rows]
answers = [row["doctor_response"] for row in rows]
documents = [f"Symptoms: {', '.join(row['symptom'])}. Disease: {row['disease']}. Treatment: {row['treatment']}." for row in rows]

# Set up PUPVectorStore
vector_store = PUPVectorStore(PUPVectorStoreConfig(top_p=0.02, epsilon=0.2))

# Add documents to the store
for doc in documents:
    vector_store.add(doc)

# Now retriever will be this vector_store
retriever = vector_store

generator = Model()

# Initialize DPSparseVoteRAGEngine
engine = DPSparseVoteRAGEngine(
    retriever=retriever,
    generator=generator,
    retrieval_epsilon=retrieval_epsilon,
    generation_epsilon=generation_epsilon,
    threshold=threshold
)

# Evaluate
results = []
correct = 0
for idx, query in enumerate(questions):
    print(f"\nProcessing Question {idx+1}: {query}")

    retrieved_docs = engine.retrieve(query, k=k)
    generated_answer = engine.generate(query, retrieved_docs, max_tokens=max_tokens)

    print("Generated Answer:", generated_answer)
    expected_answer = answers[idx]

    match = expected_answer.lower() in generated_answer.lower()
    correct += int(match)

    results.append({
        "query": query,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "match": match
    })

# Save evaluation.json
os.makedirs("results", exist_ok=True)
with open("results/evaluation.json", "w") as f:
    json.dump({
        "total_questions": num_questions,
        "correct_matches": correct,
        "accuracy": correct / num_questions,
        "detailed_results": results
    }, f, indent=2)

print(f"\nSaved evaluation.json with {correct}/{num_questions} correct answers.")
