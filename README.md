# Legal Document Analysis Chatbot

This project is a Streamlit-based chatbot for analyzing legal documents such as contracts, NDAs, leases, and wills. It extracts text (even from scanned PDFs using OCR), splits it into meaningful chunks, embeds it using a legal-domain embedding model, and uses a language model to answer user questions in context.

## Features

- Upload legal documents in PDF format
- Text extraction with fallback OCR for scanned PDFs
- Clause detection (e.g., termination, confidentiality, indemnity)
- Retrieval-based question answering using HuggingFace's Flan-T5
- Embedding with InLegalBERT (fallback to MiniLM if unavailable)
- Suggested legal FAQs generated from document summaries
- Context-aware conversational memory
- GPU acceleration support (PyTorch CUDA)

## Tech Stack

- Python 3.9+
- Streamlit
- LangChain
- HuggingFace Transformers
- PyTorch
- PyMuPDF (`fitz`)
- Tesseract OCR (`pytesseract`)
- HuggingFace Embeddings
- Chroma / Custom Vector Store

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/legal-chatbot.git
cd legal-chatbot
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the App**
```bash
streamlit run app.py
```

