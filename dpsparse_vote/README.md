# DPSparseVoteRAG

A simple Python implementation of the "Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy" paper, specifically the DPSparseVoteRAG variant.

## Project Overview

This project implements DPSparseVoteRAG, a differentially private retrieval-augmented generation (RAG) system that:
- Retrieves domain-specific documents.
- Checks if generated tokens rely on sensitive information.
- Smartly spends privacy budget only when necessary.

This work is inspired by the 2024 paper by Koga, Wu, and Chaudhuri.

## Installation

```bash
git clone <repo-url>
cd dp_sparse_vote_rag
pip install numpy
```

## Running Tests

```bash
python test_dp_sparse_vote_rag.py
```

This runs a simple test using toy retriever and generator models to demonstrate the DPSparseVoteRAG system.

## Project Structure

- `dp_sparse_vote_rag.py` — Main DPSparseVoteRAG engine.
- `toy_models.py` — Toy retriever and generator for basic testing.
- `test_dp_sparse_vote_rag.py` — Testing script to demonstrate functionality.

## New: DPSparseVoteRAG Implementation

We added an implementation of the DPSparseVoteRAG algorithm based on the paper "Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy" (2024).

See [dpsparse_vote/](./dpsparse_vote/) for the code and details.

## References

- "Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy" (Koga, Wu, Chaudhuri, 2024)

