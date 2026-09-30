from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.graph import ask

app = FastAPI(title="Agentic AI eBook RAG Chatbot")


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)


class Context(BaseModel):
    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    contexts: List[Context]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = ask(req.question)
    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        contexts=result["chunks"],
    )