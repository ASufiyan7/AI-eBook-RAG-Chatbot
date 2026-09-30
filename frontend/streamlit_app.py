import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="Agentic AI eBook Chatbot")
st.title("Agentic AI eBook Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_details(msg):
    conf = min(max(msg["confidence"], 0.0), 1.0)
    st.progress(conf, text=f"Confidence: {conf:.2f}")
    with st.expander("Retrieved context"):
        for c in msg["contexts"]:
            st.markdown(f"**Page {c['page']}** · score `{c['score']:.3f}`")
            st.write(c["text"])
            st.divider()


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            show_details(m)

if question := st.chat_input("Ask about the Agentic AI eBook..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(API_URL, json={"question": question}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

        st.write(data["answer"])
        msg = {
            "role": "assistant",
            "content": data["answer"],
            "confidence": data["confidence"],
            "contexts": data["contexts"],
        }
        show_details(msg)
    st.session_state.messages.append(msg)