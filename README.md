# 📚 Book Recommendation Chatbot (RAG)

**Authors:**  
Kemalettin Efe Gedik  
João Bernardo Sousa Faria  

---

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** based book recommendation chatbot.

Users can enter natural language queries (e.g. *“I like fantasy with dragons”*), and the system retrieves and recommends relevant books from a real dataset.

All recommendations are **grounded in retrieved data**, not hallucinated.

---

## System Description
- **Retriever:** Dense Passage Retrieval (DPR) + keyword filtering  
- **Generator:** Deterministic top-N ranking + optional LLM-based rewriting  
- **Interface:** Streamlit web application  
- **Evaluation:** RAGAS-style metrics  

---

## Dataset
We use the **Goodreads Books Dataset** from Kaggle:  
https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks

Used fields:
- title
- authors
- average_rating
- description

---

## Book Embeddings (Important)
The file book_embeddings.npy is not included in this repository because it is large (~34 MB).
It is generated offline using build_index.py and loaded at runtime by the retriever.
