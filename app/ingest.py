import io
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config
from app.vectorstore import upsert_chunks


def load_pages(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def build_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    for page_no, text in pages:
        for j, piece in enumerate(splitter.split_text(text)):
            if len(piece.strip()) < 50:
                continue
            chunks.append({"id": f"p{page_no}-c{j}", "text": piece, "page": page_no})
    return chunks


if __name__ == "__main__":
    pages = load_pages(config.PDF_URL)
    print(f"Loaded {len(pages)} pages")
    chunks = build_chunks(pages)
    print(f"Created {len(chunks)} chunks")
    upsert_chunks(chunks)
    print("Done: chunks stored in Pinecone")