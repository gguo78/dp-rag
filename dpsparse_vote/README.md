# DPSparseVoteRAG

This project implements a full experimental pipeline based on the paper:

> *"Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy" (Koga, Wu, Chaudhuri, 2024)*

with real LLM models, real retrieval systems, and real datasets (TriviaQA/NaturalQuestions).

---

## 📚 Project Structure

- `dp_sparse_vote_rag.py` — Upgraded DPSparseVoteRAG engine.
- `real_retriever.py` — FAISS + SentenceTransformer retriever.
- `real_generator.py` — Hugging Face LLM (OPT or LLaMA) based generator.
- `load_dataset.py` — Load TriviaQA or NaturalQuestions datasets.
- `test_dp_sparse_vote_rag_full.py` — Full experimental evaluation script.

---

## 🛠 Installation

```bash
pip install numpy datasets faiss-cpu transformers sentence-transformers
```
(Use `faiss-gpu` instead of `faiss-cpu` if you have GPU.)

---

## 📄 Running the Full Experiment

```bash
python test_dp_sparse_vote_rag_full.py
```

This will:
- Load a subset of TriviaQA dataset
- Build a FAISS retriever over documents
- Use a Hugging Face OPT-1.3B model to generate answers
- Apply DPSparseVoteRAG for private token generation
- Output several generated answers

---

## ⚙️ Requirements

- Python 3.9+
- 16GB RAM minimum (recommended)
- GPU recommended for faster model inference (OPT models)

---

## 📢 Notes

- We use simplified document corpus (e.g., answers only) for retrieval testing.
- The DPSparseVoteRAG pipeline follows Algorithm 2 in the original paper.
- This setup allows formal evaluation experiments similar to the paper.

---

## 🧠 Future Extensions

- Replace synthetic retrieval documents with full Wikipedia corpus.
- Expand testing to full TriviaQA/NQ datasets.
- Evaluate Match Accuracy, BLEU, privacy budget consumption.

---

## 📖 References

- Koga, Wu, and Chaudhuri (2024). *Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy*. [arXiv:2412.19291](https://arxiv.org/pdf/2412.04697)

---

