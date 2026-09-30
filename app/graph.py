from typing import TypedDict, List
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END
from app import config
from app.vectorstore import query

_llm = genai.Client(api_key=config.GOOGLE_API_KEY)

NOT_FOUND = "I couldn't find this in the Agentic AI eBook."

SYSTEM_PROMPT = f"""You answer questions using ONLY the provided context from the Agentic AI eBook.
Rules:
- Use only facts stated in the context. Never use outside knowledge.
- If the context does not contain the answer, reply exactly: {NOT_FOUND}
- Be concise and cite pages like (Page 4)."""


class State(TypedDict):
    question: str
    chunks: List[dict]
    confidence: float
    answer: str


def retrieve(state: State):
    chunks = query(state["question"])
    top = chunks[0]["score"] if chunks else 0.0
    return {"chunks": chunks, "confidence": round(top, 3)}


def guard(state: State):
    """Routing function: only call the LLM if retrieval looks relevant."""
    return "generate" if state["confidence"] >= config.MIN_SCORE else "refuse"


def refuse(state: State):
    return {"answer": NOT_FOUND}


def generate(state: State):
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in state["chunks"])
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
    for attempt in range(4):
        try:
            res = _llm.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=0
                ),
            )
            return {"answer": res.text.strip()}
        except Exception as e:
            if attempt == 3:
                raise
            import time
            time.sleep(2 * (attempt + 1))


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("refuse", refuse)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve", guard, {"generate": "generate", "refuse": "refuse"}
)
builder.add_edge("generate", END)
builder.add_edge("refuse", END)
rag_graph = builder.compile()


def ask(question: str) -> dict:
    return rag_graph.invoke({"question": question})