import streamlit as st

st.set_page_config(
    page_title="RAG LLM Application",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG LLM Application")

st.write("Welcome to our End-to-End RAG + LLM Project!")

st.write("This application will allow you to:")
st.write("- Upload PDF documents")
st.write("- Ask questions about the PDF")
st.write("- Get answers using an LLM")