# toy_models.py

import random

class ToyRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, k=5):
        # Randomly pick k documents
        return random.sample(self.documents, k=min(k, len(self.documents)))

class ToyGenerator:
    def __init__(self):
        self.vocabulary = ["the", "cat", "sat", "on", "mat", "dog", "<EOS>"]

    def generate_next_token(self, prompt):
        # Pick a random token from the vocabulary
        return random.choice(self.vocabulary)