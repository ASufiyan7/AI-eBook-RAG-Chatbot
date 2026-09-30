# Agentic AI eBook RAG Chatbot

A grounded Q&A chatbot built on the free [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) by Konverge.AI.

The main focus here is **strict grounding**:
- If the question is covered in the book, it answers with page-level citations (e.g., `Page 11`).
- If the question is outside the book or irrelevant, it refuses to answer instead of hallucinating.

Built with **LangGraph**, **Pinecone**, **Google Gemini**, **FastAPI**, and a **Streamlit** frontend.

---

## What's Inside

```
rag-chatbot/
├── app/
│   ├── config.py          # Settings: chunk size, similarity threshold, model names
│   ├── ingest.py          # Downloads the PDF, chunks by page, embeds, and uploads to Pinecone
│   ├── vectorstore.py     # Pinecone setup, Gemini embeddings + auto-retry on rate limits
│   ├── graph.py           # LangGraph flow: retrieve → guardrail check → generate / refuse
│   └── main.py            # FastAPI backend with /chat and /health endpoints
├── frontend/
│   └── streamlit_app.py   # Web UI showing chat, confidence bar, and retrieved source chunks
├── scripts/
│   └── run_samples.py     # Test script to run benchmark questions and export results
├── SAMPLE_QUERIES.md      # Output log from running sample queries
├── .env.example           # API key template
├── requirements.txt
└── README.md
```

---

## How It Works

Most RAG bots dump whatever context they find straight into an LLM and hope for the best. Here, we add an explicit guardrail step before generation:

1. **Ingest (One-time):**
   - The 60-page PDF is downloaded and split page-by-page into 137 overlapping chunks (800 chars, 150 overlap).
   - Each chunk gets embedded with `gemini-embedding-001` (768 dimensions) and stored in Pinecone with metadata (`page` and `text`).
   - Chunks use deterministic IDs (`p{page}-c{index}`), so re-running ingestion updates existing vectors instead of duplicating them.

2. **Retrieve:**
   - The user's question is embedded and Pinecone returns the top 5 most similar chunks by cosine score.

3. **Guardrail Check:**
   - We check the top chunk's similarity score against `MIN_SCORE` (`0.55`).
   - If the score is below `0.55`, the query is flagged as out-of-scope. The LLM is never called, saving time and API tokens. It immediately returns:  
     `"I couldn't find this in the Agentic AI eBook."`

4. **Generate:**
   - If the score passes, the chunks are passed to `gemini-3.5-flash` at `temperature=0` with a strict system prompt telling it to answer only from the provided text and include page citations.

---

## Quickstart

### 1. Requirements

- Python 3.10 or higher
- A free [Google AI Studio API key](https://aistudio.google.com/)
- A free [Pinecone API key](https://app.pinecone.io/)

### 2. Install

```bash
git clone <your-repo-url>
cd rag-chatbot

# Set up virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Or on macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set API Keys

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GOOGLE_API_KEY="your-google-api-key-here"
PINECONE_API_KEY="your-pinecone-api-key-here"
```

### 4. Ingest the eBook (Run Once)

```bash
python -m app.ingest
```

This pulls the PDF from the web, processes all 60 pages into 137 chunks, and uploads them to Pinecone. The index (`agentic-ai-ebook`) is created automatically if it doesn't already exist.

> **Note:** The script includes automatic retry logic with backoff, so it gracefully pauses and resumes if you hit the Google free-tier per-minute embedding limit.

### 5. Run the Backend API

```bash
uvicorn app.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`. You can test endpoints via the built-in Swagger UI at `http://127.0.0.1:8000/docs`.

### 6. Run the Frontend (Streamlit)

In a separate terminal (with `.venv` activated and inside the `rag-chatbot` directory):

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501` in your browser. You get an interactive chat interface that displays:
- The generated answer with page numbers.
- A confidence bar (top similarity score).
- An expandable section showing the exact source text and page numbers pulled from the eBook.

---

## Example Queries & Behavior

Here is how the bot handles in-scope vs. out-of-scope questions:

### In-scope Question
> **User:** What are the key components of an agentic AI system?  
> **Confidence:** `0.81`  
> **Bot:** *Perception, Reasoning, Planning, Learning, and Execution (Page 17).*

### Out-of-scope Question
> **User:** Who won the 2023 Cricket World Cup?  
> **Confidence:** `0.49` *(below 0.55 cutoff)*  
> **Bot:** *I couldn't find this in the Agentic AI eBook.*

You can run the full test suite anytime with:

```bash
python -m scripts.run_samples
```

Results are saved to [`SAMPLE_QUERIES.md`](SAMPLE_QUERIES.md).

---

## API Reference

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is agentic AI?"}'
```

**Response:**

```json
{
  "answer": "Based on the provided context, Agentic AI is an autonomous system that creates impact by learning continuously, focusing on goals, and acting independently...",
  "confidence": 0.78,
  "contexts": [
    {
      "text": "Agentic AI goes beyond other AI systems...",
      "page": 11,
      "score": 0.78
    }
  ]
}
```

### `GET /health`

Health check endpoint. Returns `{"status": "ok"}`.

---

## Configuration

Everything configurable is kept in [`app/config.py`](app/config.py):

| Variable | Default | Purpose |
|---|---|---|
| `EMBED_MODEL` | `gemini-embedding-001` | Google embedding model name |
| `EMBED_DIM` | `768` | Vector dimension size |
| `LLM_MODEL` | `gemini-3.5-flash` | LLM used for generation |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP`| `150` | Character overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved from Pinecone |
| `MIN_SCORE` | `0.55` | Similarity threshold for the guardrail node |

If you find relevant questions being refused, lower `MIN_SCORE` to `0.50`. If off-topic queries sneak through, bump it to `0.60`.