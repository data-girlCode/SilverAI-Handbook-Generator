import streamlit as st
import os
from dotenv import load_dotenv

# Import our modules from src
from src.pdf_processor import process_pdf
from src.rag_pipeline import query_knowledge_graph
from src.chat_handler import get_chat_response
from src.handbook_gen import generate_handbook

# Load environment variables
load_dotenv()

st.set_page_config(page_title="SilverAI Handbook Generator", page_icon="📖", layout="wide")

st.title("SilverAI Handbook Generator")
st.markdown("> An AI-powered chat application that transforms your PDF documents into comprehensive, 20,000-word handbooks.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading PDFs
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing and indexing PDFs..."):
                st.success("PDFs processed successfully!")
        else:
            st.warning("Please upload some PDFs first.")
            
    st.divider()
    
    st.header("Generate Handbook")
    topic = st.text_input("Handbook Topic", placeholder="e.g. Retrieval-Augmented Generation")
    if st.button("Generate 20k Word Handbook"):
        if topic:
            with st.spinner("Generating your comprehensive handbook... This may take a while. ⏳"):
                st.success("Handbook generated!")
        else:
            st.warning("Please enter a topic.")

# Main Chat Interface
st.header("Chat with your Documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Placeholder for actual RAG logic
        response = f"I am a placeholder response for: {prompt}"
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
