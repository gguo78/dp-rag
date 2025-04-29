# DPSparseVoteRAG Integration

This folder implements the **DPSparseVoteRAG** algorithm described in the paper:

> "Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy" (Koga, Wu, and Chaudhuri, 2024)

The DPSparseVoteRAG engine integrates into our existing RAG pipeline using:
- The same `PupRetriever` retriever
- The same `SimpleGenerator` generator
- Our current dataset

Privacy budget is spent only when necessary using a noisy threshold comparison against non-private token outputs, as described in Algorithm 2 of the paper.

---

## Files

- `dp_sparse_vote_rag_engine.py` — DPSparseVoteRAG engine compatible with the group's retriever and generator.
- `test_dp_sparse_vote_rag.py` — Test script that evaluates DPSparseVoteRAG and saves results to `results/evaluation.json`.

---

## How to Run

```bash
python test_dp_sparse_vote_rag.py
