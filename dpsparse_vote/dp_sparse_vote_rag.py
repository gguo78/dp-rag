# dp_sparse_vote_rag_v2.py

import random
import numpy as np
from collections import Counter

class DPSparseVoteRAG:
    def __init__(self, retriever, generator, 
                 num_voters=10, k=1, 
                 epsilon_total=10, delta_total=1e-5, threshold=0.5):
        self.retriever = retriever
        self.generator = generator
        self.num_voters = num_voters
        self.k = k
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        self.threshold = threshold

        # Split privacy budget: half for threshold testing, half for private voting
        self.epsilon_per_threshold = epsilon_total / 2
        self.epsilon_per_vote = epsilon_total / 2

        # Budget counters
        self.max_private_votes = int(np.floor(epsilon_total / self.epsilon_per_vote))
        self.remaining_private_votes = self.max_private_votes

        # Initialize noisy threshold tau_hat
        self.tau_hat = self.threshold + np.random.laplace(scale=2/self.epsilon_per_threshold)

    def retrieve_documents(self, query):
        docs = self.retriever.retrieve(query, k=self.num_voters * self.k)
        partitions = [docs[i::self.num_voters] for i in range(self.num_voters)]
        return partitions

    def build_prompt(self, query, previous_tokens, docs):
        # Simple prompt construction
        return query + " " + " ".join(previous_tokens) + " " + " ".join(docs)

    def generate_token(self, query, previous_tokens, docs):
        prompt = self.build_prompt(query, previous_tokens, docs)
        return self.generator.generate_next_token(prompt)

    def build_histogram(self, tokens):
        return Counter(tokens)

    def noisy_threshold_test(self, histogram, non_rag_token):
        count = histogram.get(non_rag_token, 0)
        noisy_count = count + np.random.laplace(scale=4/self.epsilon_per_threshold)
        return noisy_count >= self.tau_hat

    def limited_domain_mechanism(self, histogram):
        # Apply Laplace noise to each token count
        noisy_counts = {token: count + np.random.laplace(scale=2/self.epsilon_per_vote)
                        for token, count in histogram.items()}
        # Choose token with maximum noisy count
        return max(noisy_counts, key=noisy_counts.get)

    def generate_answer(self, query, max_tokens=50):
        answer_tokens = []
        previous_tokens = []

        # Retrieve and partition documents
        partitions = self.retrieve_documents(query)

        for t in range(max_tokens):
            if self.remaining_private_votes <= 0:
                break  # Stop if no privacy budget left

            # Generate non-RAG token
            non_rag_token = self.generate_token(query, previous_tokens, docs=[])

            # Generate RAG tokens from each voter
            rag_tokens = [self.generate_token(query, previous_tokens, partition) 
                          for partition in partitions]

            histogram = self.build_histogram(rag_tokens)

            # Noisy threshold test
            if self.noisy_threshold_test(histogram, non_rag_token):
                token = non_rag_token
                # No privacy budget consumed
            else:
                token = self.limited_domain_mechanism(histogram)
                # Consume privacy budget only when private voting happens
                self.remaining_private_votes -= 1
                # Update noisy threshold for next token
                self.tau_hat = self.threshold + np.random.laplace(scale=2/self.epsilon_per_threshold)

            answer_tokens.append(token)
            previous_tokens.append(token)

            if token == "<EOS>":
                break

        return " ".join(answer_tokens)