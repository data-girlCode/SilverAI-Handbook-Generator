# SilverAI-Handbook-Generator

# 📖 SilverAI Handbook Generator

> An AI-powered chat application that transforms your PDF documents into comprehensive, 20,000-word handbooks — all through a simple conversational interface. Access here: https://silverai-handbook-generator.streamlit.app/

---

## ✨ Features

### 📄 PDF Upload & Ingestion
- Upload one or multiple PDF documents through the chat interface
- Automatic text extraction and chunking using **PyPDF** / **pdfplumber**
- Content is instantly indexed and ready for querying

### 🧠 AI-Powered Knowledge Graph (LightRAG + Supabase)
- Uploaded PDFs are processed into a **knowledge graph** using [LightRAG](https://github.com/HKUDS/LightRAG)
- Embeddings stored in **Supabase** with `pgvector` for fast semantic retrieval
- Relationships between concepts are preserved for richer, context-aware responses

### 💬 Contextual Chat Interface
- Ask any question about your uploaded documents
- Responses are grounded in the content of your PDFs — not generic AI answers
- Clean, minimal chat UI built with **Gradio** / **Streamlit**

### 📚 20,000-Word Handbook Generation
- Request a full handbook with a single chat message, e.g.:
  > *"Generate a comprehensive handbook on Retrieval-Augmented Generation"*
- Powered by **Grok 4.1** using the **LongWriter** technique for ultra-long structured output
- Output includes:
  - Table of Contents
  - Structured chapters and headings
  - Detailed sections derived from your uploaded materials
  - In-text citations referencing your source PDFs


## 🎬 Demo

### Upload & Chat
```
User:  [Uploads: rag-paper.pdf, llm-survey.pdf]

User:  What are the main components of a RAG system?

AI:    Based on your uploaded documents, a RAG system consists of three 
       core components: a retriever, a knowledge store, and a generator...
```

### Handbook Generation
```
User:  Create a handbook on Retrieval-Augmented Generation

AI:    Generating your 20,000-word handbook... ⏳

       Done! Here's your handbook:

       # The Complete Guide to Retrieval-Augmented Generation
       
       ## Table of Contents
       1. Introduction to RAG .............. 3
       2. Core Architecture ................ 8
       3. Retrieval Strategies ............. 15
       ...
       [20,000+ words]
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SilverAI Handbook Generator                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📄 PDF Upload  →  🧠 LightRAG  →  💬 Chat UI  →  📖 Handbook │
│                      (Supabase)     (Grok 4.1)   (20k words) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Gradio / Streamlit | Chat interface & PDF upload |
| **LLM** | Grok 4.1 | Long-context generation (LongWriter) |
| **RAG** | LightRAG | Knowledge graph from PDFs |
| **Vector DB** | Supabase (pgvector) | Embedding storage & retrieval |
| **PDF Processing** | PyPDF / pdfplumber | Text extraction |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Supabase account & project URL
- Grok API key (xAI)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SilverAI-Handbook-Generator.git
cd SilverAI-Handbook-Generator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open your browser at `http://localhost:7860` (Gradio) or `http://localhost:8501` (Streamlit).

---

## 📁 Project Structure

```
SilverAI-Handbook-Generator/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── ingestion/
│   ├── pdf_parser.py       # PDF text extraction
│   └── chunker.py          # Text chunking logic
│
├── rag/
│   ├── lightrag_setup.py   # LightRAG knowledge graph setup
│   └── retriever.py        # Query & retrieval logic
│
├── generation/
│   ├── grok_client.py      # Grok 4.1 API integration
│   └── longwriter.py       # LongWriter handbook generation
│
├── ui/
│   └── chat_interface.py   # Gradio / Streamlit chat UI
│
├── LongWriter-main/        # Reference LongWriter implementation
└── Documentation/          # Research paper & assignment brief
```

---

## 🧪 Test It Yourself

1. Upload 2–3 AI-related PDFs (e.g., research papers on RAG, LLMs, or vector databases)
2. Ask a question: *"What is the difference between sparse and dense retrieval?"*
3. Generate a handbook: *"Create a handbook on large language models"*
4. Receive a 20,000+ word structured document grounded in your uploads


---

## 📄 License

This project was built as part of the **SilverAI AI Engineering Assignment** by LunarTech.  
All provided materials are the intellectual property of LunarTech and are for evaluation purposes only.
