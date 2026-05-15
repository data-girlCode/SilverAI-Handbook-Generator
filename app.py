from src.chat_handler import get_chat_response
from src.handbook_gen import generate_handbook
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from src.pdf_processor import process_pdf
from src.rag_pipeline import initialize, ingest_chunks, query_knowledge_graph

load_dotenv()

# Grok client
grok = OpenAI(
    api_key=os.getenv("GROK_API_KEY") or st.secrets.get("GROK_API_KEY", ""),
    base_url="https://api.x.ai/v1"
)

# Initialize LightRAG once at startup
if "rag_initialized" not in st.session_state:
    initialize()
    st.session_state.rag_initialized = True

# Page config
st.set_page_config(page_title="SilverAI Handbook Generator", page_icon="📖")
st.title("📖 SilverAI Handbook Generator")
st.caption("Upload PDFs, ask questions, and generate a 20,000-word handbook.")

# --- PDF Upload ---
st.subheader("📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = f"uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner(f"Processing {uploaded_file.name}..."):
            chunks = process_pdf(save_path)
            ingest_chunks(chunks, uploaded_file.name)
        st.success(f"✅ {uploaded_file.name} indexed!")

st.divider()

# --- Chat Interface ---
st.subheader("💬 Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Upload a PDF and ask me anything about it, or ask me to generate a handbook. I can also just chat!"
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Try to get RAG context if documents have been uploaded
            rag_context = ""
            try:
                results = query_knowledge_graph(prompt)
                if results and results[0]:
                    rag_context = results[0]
            except Exception:
                pass

            # Build system prompt
            if rag_context:
                system_prompt = f"""You are SilverAI, a helpful AI assistant. 
You have access to the following context from uploaded documents:

{rag_context}

Use this context to answer the user's question accurately. 
If asked to generate a handbook, write a comprehensive, structured document with a table of contents, 
headings, and detailed sections based on the uploaded content.
If the question is unrelated to the documents, just answer conversationally."""
            else:
                system_prompt = """You are SilverAI, a helpful AI assistant. 
Answer the user's questions conversationally. 
If they ask about documents, remind them to upload a PDF first."""

            # Build message history for Grok (last 10 messages for context)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-10:]
            ]

            # Call Grok API
            try:
                response = grok.chat.completions.create(
                    model="grok-3",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *history
                    ],
                    max_tokens=4096,
                    temperature=0.7
                )
                reply = response.choices[0].message.content

            except Exception as e:
                reply = f"⚠️ Error calling Grok API: {str(e)}\n\nCheck that your GROK_API_KEY in .env is valid."

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})