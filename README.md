# RAG PDF Chat

A web app that lets you upload any PDF and ask questions about it using Retrieval-Augmented Generation (RAG) and Claude AI.

## How it works

1. Upload a PDF via the browser UI
2. The app extracts text using PyPDF2 and splits it into chunks
3. When you ask a question, TF-IDF similarity finds the most relevant chunks
4. Only those chunks are sent to Claude — not the whole document
5. Claude answers based on the retrieved context

This is a minimal RAG implementation — the core technique used in production AI search and Q&A products.

## Demo

![PDF Q&A UI](https://via.placeholder.com/700x400?text=Upload+PDF+%E2%86%92+Ask+Questions)

Visit `http://localhost:8080`, upload a PDF, and start asking questions.

## Setup

```bash
git clone https://github.com/manoj39tammali/rag-pdf-chat.git
cd rag-pdf-chat
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key at [console.anthropic.com](https://console.anthropic.com).

## Usage

```bash
python app.py
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

## Tech Stack

- [Anthropic Claude](https://anthropic.com) — AI question answering (claude-haiku-4-5)
- [Flask](https://flask.palletsprojects.com) — web server
- [PyPDF2](https://pypdf2.readthedocs.io) — PDF text extraction
- [scikit-learn](https://scikit-learn.org) — TF-IDF similarity for chunk retrieval
- [python-dotenv](https://github.com/theskumar/python-dotenv) — environment variables

## Topics

`python` `ai` `claude` `anthropic` `rag` `retrieval-augmented-generation` `pdf` `flask` `llm` `nlp`
