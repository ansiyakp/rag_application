# 📚 PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions using information retrieved from PDF documents.

## 🚀 Live Application

👉 **[Open the PDF RAG Chatbot](https://ragapplication-h4n9dgd5cstevvbgsnkrjh.streamlit.app/)**

## ✨ Features

* 📄 PDF-based question answering
* 🔎 Semantic similarity search
* 🗄️ Qdrant Cloud vector database
* 🤖 Groq cloud LLM
* 🧠 SentenceTransformer embeddings
* 🛡️ Hallucination protection
* 📑 PDF and page citations
* ☁️ Streamlit Cloud deployment

## 🛠️ Technologies

* Python
* Streamlit
* Qdrant Cloud
* Sentence Transformers
* Groq
* PyMuPDF

## 🏗️ Architecture

```text
PDF Documents
      ↓
   ingest.py
      ↓
SentenceTransformer
      ↓
 Qdrant Cloud
      ↓
    rag.py
      ↓
  Groq Cloud
      ↓
   Streamlit
      ↓
🌐 Online Chatbot
```

## 📂 Project Structure

```text
rag_application/
├── app.py
├── rag.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
```

## 🔐 Environment Variables

The application uses:

```text
QDRANT_URL
QDRANT_API_KEY
GROQ_API_KEY
```

API keys are stored securely using environment variables and Streamlit Secrets. They are **not included in this repository**.

## 🎯 Purpose

This project demonstrates how to build and deploy a cloud-based **PDF RAG chatbot** using vector search, embeddings, and a cloud LLM.

## 👩‍💻 Author

**Ansiya KP**

