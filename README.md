# AI PDF Chatbot using RAG and Ollama

## Project Overview

This project is an AI-powered PDF chatbot that allows users to upload a PDF document and ask questions about its contents.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the PDF and generate natural-language answers using the Qwen 2.5 language model running locally through Ollama.

## Technologies Used

- Python — Programming language
- Streamlit — Web interface
- pypdf — PDF text extraction
- scikit-learn — TF-IDF and cosine similarity for retrieval
- Ollama — Runs AI models locally
- Qwen 2.5 (3B) — Large Language Model (LLM)

## How It Works

1. The user uploads a PDF.
2. Text is extracted from the PDF using pypdf.
3. The extracted text is divided into overlapping chunks.
4. TF-IDF and cosine similarity identify relevant chunks.
5. The relevant chunks and user question are sent to Qwen.
6. Qwen generates an answer based on the retrieved information.

## Installation

Install Python and Ollama before proceeding.

Clone or download the GitHub repository and open its folder in a terminal.

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Download the AI model:

```bash
ollama pull qwen2.5:3b
```

Run the chatbot:

```bash
python -m streamlit run app.py
```

Open the local Streamlit URL in your browser.

## Features

- PDF upload and text extraction
- Question answering using a local LLM
- TF-IDF-based document retrieval
- Overlapping text chunks
- No paid API key required
- Option to inspect retrieved PDF chunks

## Limitations

- Scanned PDFs without extractable text are not supported.
- Retrieval may miss relevant information.
- The LLM may occasionally generate incorrect answers.
- Ollama and the Qwen model must be installed separately.

## Future Improvements

- Semantic embeddings and vector database retrieval
- Support for multiple PDFs
- Source citations with page numbers
- Improved answer validation
