# dp_sparse_vote_rag.py

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

        self.epsilon_per_token = epsilon_total / 2
        self.epsilon_per_threshold = epsilon_total / 2
        self.cmax = int(np.floor(epsilon_total / self.epsilon_per_token))
        self.remaining_budget = self.cmax

    def retrieve_documents(self, query):
        docs = self.retriever.retrieve(query, k=self.num_voters * self.k)
        partitions = [docs[i::self.num_voters] for i in range(self.num_voters)]
        return partitions

    def generate_token(self, query, previous_tokens, docs):
        prompt = self.build_prompt(query, previous_tokens, docs)
        return self.generator.generate_next_token(prompt)

    def build_prompt(self, query, previous_tokens, docs):
        return query + " " + " ".join(previous_tokens) + " " + " ".join(docs)

    def build_histogram(self, tokens):
        return Counter(tokens)

    def noisy_threshold_test(self, histogram, non_rag_token):
        count = histogram.get(non_rag_token, 0)
        noisy_count = count + np.random.laplace(scale=2/self.epsilon_per_threshold)
        return noisy_count > self.threshold

    def limited_domain_mechanism(self, histogram):
        noisy_counts = {token: count + np.random.laplace(scale=2/self.epsilon_per_token)
                        for token, count in histogram.items()}
        return max(noisy_counts, key=noisy_counts.get)

    def generate_answer(self, query):
        answer_tokens = []
        previous_tokens = []

        partitions = self.retrieve_documents(query)

        for t in range(self.cmax):
            non_rag_token = self.generate_token(query, previous_tokens, docs=[])

            rag_tokens = [self.generate_token(query, previous_tokens, partition) 
                          for partition in partitions]

            histogram = self.build_histogram(rag_tokens)

            if self.noisy_threshold_test(histogram, non_rag_token):
                token = non_rag_token
            else:
                token = self.limited_domain_mechanism(histogram)
                self.remaining_budget -= 1

            answer_tokens.append(token)
            previous_tokens.append(token)

            if token == "<EOS>" or self.remaining_budget <= 0:
                break

        return " ".join(answer_tokens)
