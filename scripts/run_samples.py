from pathlib import Path
from app.graph import ask

QUESTIONS = [
    "What is agentic AI?",
    "How does agentic AI differ from traditional AI or generative AI?",
    "What are the key components of an agentic AI system?",
    "What are the main business benefits of agentic AI?",
    "What challenges or risks are associated with agentic AI?",
    "Who won the 2023 Cricket World Cup?"
]

lines = ["# Sample Queries\n"]
for q in QUESTIONS:
    r = ask(q)
    pages = sorted({c["page"] for c in r["chunks"]})
    lines += [
        f"## Q: {q}\n",
        f"**Answer:** {r['answer']}\n",
        f"**Confidence:** {r['confidence']}\n",
        f"**Retrieved from pages:** {pages}\n",
    ]

Path("SAMPLE_QUERIES.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved SAMPLE_QUERIES.md")