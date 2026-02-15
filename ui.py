import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="AIRMAN Aviation RAG", layout="wide")

st.title("✈️ AIRMAN Aviation Document Assistant")
st.markdown("Ask questions strictly from the aviation manuals.")

# Debug toggle
debug_mode = st.checkbox("Enable Debug Mode (Show Retrieved Chunks)")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={"question": question, "debug": debug_mode}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("📘 Answer")
            st.write(data["answer"])

            st.subheader("📎 Citations")
            for c in data["citations"]:
                st.write(f"- {c['document']} (Page {c['page']})")

            if debug_mode and "retrieved_chunks" in data:
                st.subheader("🔍 Retrieved Chunks")
                for chunk in data["retrieved_chunks"]:
                    st.text(chunk["text"][:1000])
        else:
            st.error("Error connecting to backend.")
