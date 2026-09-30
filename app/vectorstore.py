import re
import time
from google import genai
from google.genai import types, errors
from pinecone import Pinecone, ServerlessSpec
from app import config

_llm = genai.Client(api_key=config.GOOGLE_API_KEY)
_pc = Pinecone(api_key=config.PINECONE_API_KEY)


def embed(texts, task, max_retries=6):
    """task: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions."""
    for attempt in range(max_retries):
        try:
            res = _llm.models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=config.EMBED_DIM
                ),
            )
            return [e.values for e in res.embeddings]
        except (errors.APIError, Exception) as e:
            if attempt == max_retries - 1:
                raise
            delay = 2 ** attempt * 5
            err_msg = str(e)
            match = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err_msg)
            if match:
                delay = float(match.group(1)) + 2.0
            status = getattr(e, "code", "429")
            print(f"Rate limited or API error ({status}). Waiting {delay:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


def get_index():
    if config.INDEX_NAME not in _pc.list_indexes().names():
        _pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return _pc.Index(config.INDEX_NAME)


def upsert_chunks(chunks, batch=25):
    """chunks: list of {'id', 'text', 'page'}"""
    index = get_index()
    total = len(chunks)
    for i in range(0, total, batch):
        part = chunks[i : i + batch]
        print(f"Processing chunks {i + 1} to {min(i + batch, total)} of {total}...")
        vectors = embed([c["text"] for c in part], "RETRIEVAL_DOCUMENT")
        index.upsert(
            vectors=[
                {"id": c["id"], "values": v,
                 "metadata": {"text": c["text"], "page": c["page"]}}
                for c, v in zip(part, vectors)
            ]
        )
        time.sleep(1)


def query(question, k=config.TOP_K):
    vec = embed([question], "RETRIEVAL_QUERY")[0]
    res = get_index().query(vector=vec, top_k=k, include_metadata=True)
    return [
        {"text": m.metadata["text"], "page": int(m.metadata["page"]), "score": m.score}
        for m in res.matches
    ]
