from cffi import api
import time
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from app import config

_llm = genai.Client(api_key=config.GOOGLE_API_KEY)
_pc = Pinecone(api_key=config.PINECONE_API_KEY)

def embed(texts, task):
    """task: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions."""
    res = _llm.models.embed_content(
        model=config.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task, output_dimensionality=config.EMBED_DIM
        ),
    )
    return [e.values for e in res.embeddings]


def get_index():
    if config.INDEX_NAME not in _pc.list_indexes().names():
        _pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return _pc.Index(config.INDEX_NAME)


def upsert_chunks(chunks, batch=50):
    """chunks: list of {'id', 'text', 'page'}"""
    index = get_index()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
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
