# dp_sparse_vote_rag_engine.py

import random
import numpy as np
from collections import Counter

class DPSparseVoteRAGEngine:
    def __init__(self, retriever, generator, retrieval_epsilon=1.0, generation_epsilon=1.0, threshold=0.5):
        self.retriever = retriever
        self.generator = generator
        self.retrieval_epsilon = retrieval_epsilon
        self.generation_epsilon = generation_epsilon
        self.threshold = threshold

        # Privacy budget split: half for threshold, half for private voting
        self.epsilon_per_threshold = generation_epsilon / 2
        self.epsilon_per_vote = generation_epsilon / 2

        # Budget counters
        self.max_private_votes = int(np.floor(generation_epsilon / self.epsilon_per_vote))
        self.remaining_private_votes = self.max_private_votes

        # Initialize noisy threshold tau_hat
        self.tau_hat = self.threshold + np.random.laplace(scale=2/self.epsilon_per_threshold)

    def retrieve(self, query, k=5):
        docs = self.retriever.retrieve(query, k=k)
        partitions = [docs[i::k] for i in range(k)]
        return partitions

    def build_prompt(self, query, previous_tokens, docs):
        return query + " " + " ".join(previous_tokens) + " " + " ".join(docs)

    def generate_token(self, query, previous_tokens, docs):
        prompt = self.build_prompt(query, previous_tokens, docs)
        return self.generator.generate(prompt)

    def build_histogram(self, tokens):
        return Counter(tokens)

    def noisy_threshold_test(self, histogram, non_rag_token):
        count = histogram.get(non_rag_token, 0)
        noisy_count = count + np.random.laplace(scale=4/self.epsilon_per_threshold)
        return noisy_count >= self.tau_hat

    def limited_domain_mechanism(self, histogram):
        noisy_counts = {token: count + np.random.laplace(scale=2/self.epsilon_per_vote)
                        for token, count in histogram.items()}
        return max(noisy_counts, key=noisy_counts.get)

    def generate(self, query, retrieved_docs, max_tokens=50):
        answer_tokens = []
        previous_tokens = []

        partitions = retrieved_docs

        for t in range(max_tokens):
            if self.remaining_private_votes <= 0:
                break

            non_rag_token = self.generate_token(query, previous_tokens, docs=[])

            rag_tokens = [self.generate_token(query, previous_tokens, docs) for docs in partitions]

            histogram = self.build_histogram(rag_tokens)

            if self.noisy_threshold_test(histogram, non_rag_token):
                token = non_rag_token
            else:
                token = self.limited_domain_mechanism(histogram)
                self.remaining_private_votes -= 1
                self.tau_hat = self.threshold + np.random.laplace(scale=2/self.epsilon_per_threshold)

            answer_tokens.append(token)
            previous_tokens.append(token)

            if token == "<EOS>":
                break

        return " ".join(answer_tokens)
