# Agentic AI eBook RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions **strictly from the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)**. Built with LangGraph, Pinecone, Gemini, FastAPI, and a Streamlit UI.

Every response returns the final answer, the retrieved context chunks (with page numbers and similarity scores), and a confidence score. Questions the eBook cannot answer are refused instead of guessed.

## Architecture

```
                  ┌─────────────── Ingestion (one-time) ───────────────┐
                  │ PDF → extract text per page → chunk (800/150) →    │
                  │ Gemini embeddings → Pinecone (text + page metadata)│
                  └────────────────────────────────────────────────────┘

Streamlit UI ─▶ FastAPI /chat ─▶ LangGraph
 (or any client)                    │
                               [retrieve]  embed question, top-5 chunks from Pinecone
                                    │
                                 (guard)   top score ≥ MIN_SCORE ?
                                  │     │
                                yes     no
                                  │     │
                            [generate] [refuse]   "Not found in the eBook"
                         Gemini, context-only
                                  │
                                  ▼
                  { answer, confidence, contexts[] }
```

**Components**

| Component | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit graph with a conditional edge for the grounding guard |
| Vector DB | Pinecone (serverless, free tier) | Managed, cosine similarity, metadata storage |
| Embeddings | `gemini-embedding-001` (768 dims) | Free tier; separate task types for documents and queries |
| LLM | `gemini-2.5-flash` (temperature 0) | Free tier, fast, deterministic |
| API | FastAPI | Typed responses, auto-generated Swagger UI at `/docs` |
| UI | Streamlit | Chat interface that calls the API; shows confidence and retrieved chunks |

**How answers stay grounded**

1. **Retrieval guard:** if the best chunk's similarity score is below `MIN_SCORE`, the LLM is never called and the bot replies that the answer isn't in the eBook.
2. **Prompt constraint:** the LLM is told to use only the supplied context and to reply with a fixed "not found" message otherwise. It cites pages, e.g. `(Page 4)`.
3. **Temperature 0:** keeps output deterministic and close to the source text.

**Confidence score:** the cosine similarity of the top retrieved chunk (0 to 1). The same value drives the guard.

**Design notes**

- Chunking is done per page, so every chunk keeps an accurate page number for citations.
- Chunk IDs are deterministic (`p{page}-c{index}`), so re-running ingestion overwrites rather than duplicates.
- The Streamlit UI is a thin client over the API, so there is a single source of truth for the RAG logic.

## Project structure

```
rag-chatbot/
├── app/
│   ├── config.py          # env vars and tunables (chunk size, top-k, threshold)
│   ├── vectorstore.py     # embeddings + Pinecone init / upsert / query
│   ├── ingest.py          # PDF → chunks → embeddings → Pinecone
│   ├── graph.py           # LangGraph: retrieve → guard → generate / refuse
│   └── main.py            # FastAPI app (/chat, /health)
├── frontend/
│   └── streamlit_app.py   # Streamlit chat UI (calls the API)
├── scripts/
│   └── run_samples.py     # runs sample queries, writes SAMPLE_QUERIES.md
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+, a free [Google AI Studio](https://aistudio.google.com) API key, and a free [Pinecone](https://app.pinecone.io) API key.

```bash
git clone <your-repo-url>
cd rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then add GOOGLE_API_KEY and PINECONE_API_KEY
```

**1. Ingest the PDF (once)**

```bash
python -m app.ingest
```

This downloads the eBook, chunks it, embeds the chunks, and stores them in Pinecone. The index is created automatically.

**2. Start the API**

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> to try the chatbot from the Swagger UI, or use curl:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is agentic AI?"}'
```

**3. (Optional) Start the Streamlit UI**

With the API running, in a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Opens at <http://localhost:8501> with a chat interface showing the answer, a confidence bar, and the retrieved chunks. To point the UI at a different API address, set the `API_URL` environment variable (default: `http://127.0.0.1:8000/chat`).

## API

**`POST /chat`**

Request:

```json
{ "question": "What is agentic AI?" }
```

Response:

```json
{
  "answer": "...",
  "confidence": 0.0,
  "contexts": [
    { "text": "...", "page": 0, "score": 0.0 }
  ]
}
```

- `answer`: grounded answer with page citations, or a "not found" message.
- `confidence`: similarity of the best retrieved chunk (0 to 1).
- `contexts`: the top-5 retrieved chunks with page number and score.

**`GET /health`** returns `{"status": "ok"}`.

## Sample queries

Run all of them with:

```bash
python -m scripts.run_samples
```

Results are written to [`SAMPLE_QUERIES.md`](SAMPLE_QUERIES.md).

1. What is agentic AI?
2. How does agentic AI differ from traditional AI or generative AI?
3. What are the key components of an agentic AI system?
4. What are the main business benefits of agentic AI?
5. What challenges or risks are associated with agentic AI?
6. Who won the 2018 FIFA World Cup? *(out of scope, expected to be refused)*

## Configuration

All tunables live in `app/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Characters per chunk / overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per question |
| `MIN_SCORE` | 0.55 | Minimum top similarity to attempt an answer |

If valid questions get refused, lower `MIN_SCORE`. If off-topic questions get answered, raise it.

## Limitations

- Text-only extraction: tables and images in the PDF are not interpreted.
- Single-turn: each question is answered independently, with no chat memory (the UI shows history, but the backend does not use it).
- Confidence is a retrieval-similarity signal, not a guarantee of answer correctness.