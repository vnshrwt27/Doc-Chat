import streamlit as st
import tempfile
from base import BaseHandler

st.set_page_config(page_title="📚 Ingest & Chat", layout="wide")
st.title("📚 Ingest PDFs & Chat with LangChain + Groq + Pinecone")

# Sidebar - Model config
with st.sidebar:
    st.header("Model Configuration")
    selected_model = st.text_input("Model", value="meta-llama/llama-4-scout-17b-16e-instruct")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.6)

# Initialize handler
handler = BaseHandler(chat_model=selected_model, temperature=temperature)

# Section 1: PDF Upload
st.subheader("1. Upload and Ingest PDF")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("Ingesting file...")
    try:
        docs = handler.load_data(tmp_path)
        handler.ingest_data(docs)
        st.success("✅ Document successfully ingested into Pinecone.")
    except Exception as e:
        st.error(f"❌ Ingestion failed: {e}")

# Section 2: Chat Interface
st.subheader("2. Chat with Your Data")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_query = st.chat_input("Ask a question...")

if user_query:
    st.chat_message("user").markdown(user_query)

    # Format chat history as (user, assistant) pairs
    history_pairs = []
    msgs = st.session_state.chat_history
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            history_pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))

    try:
        result = handler.chat(query=user_query, chat_history=history_pairs)
        response = result['answer'] if isinstance(result, dict) and 'answer' in result else str(result)
        st.chat_message("assistant").markdown(response)

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"❌ Chat failed: {e}")
