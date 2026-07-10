import hashlib
import os
from io import BytesIO

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Load variables from the .env file.
load_dotenv()

# Read the OpenAI API key from .env.
api_key = os.getenv("OPENAI_API_KEY")


# Configure the Streamlit browser page.
st.set_page_config(
    page_title="RAG LLM Application",
    page_icon="🤖",
    layout="wide",
)


# Create a ChromaDB client once and keep it during the session.
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

# Store whether the PDF has been processed.
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False


st.title("🤖 RAG LLM Application")

st.write(
    "Upload a PDF, process the document, and ask questions about its content."
)


# Stop the application if the API key is missing.
if not api_key:
    st.error(
        "OpenAI API key was not found. Add "
        "OPENAI_API_KEY=your_key to the .env file."
    )
    st.stop()


# Create the PDF upload box.
uploaded_file = st.file_uploader(
    "📄 Upload your PDF file",
    type=["pdf"],
)


if uploaded_file is not None:
    st.success(f"Uploaded successfully: {uploaded_file.name}")

    # Get the complete uploaded PDF as bytes.
    pdf_bytes = uploaded_file.getvalue()

    # Create a unique ID from the PDF contents.
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    collection_name = f"pdf_{document_hash}"

    # Process the PDF only after the user clicks this button.
    if st.button("Process PDF"):
        try:
            with st.spinner("Reading and processing the PDF..."):
                # Open the PDF from memory.
                pdf_reader = PdfReader(BytesIO(pdf_bytes))

                # Extract text from every page.
                extracted_text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_text += page_text + "\n"

                # Check whether usable text was found.
                if not extracted_text.strip():
                    st.error(
                        "No text was found in this PDF. "
                        "It may be a scanned image PDF."
                    )
                    st.stop()

                # Divide the PDF text into smaller overlapping chunks.
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                chunks = text_splitter.split_text(extracted_text)

                # Convert each text chunk into a numerical embedding.
                embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key,
                )

                chunk_embeddings = embedding_model.embed_documents(chunks)

                # Create or open a ChromaDB collection for this PDF.
                collection = (
                    st.session_state.chroma_client.get_or_create_collection(
                        name=collection_name
                    )
                )

                # Create a unique ID for every chunk.
                chunk_ids = [
                    f"{document_hash}chunk{index}"
                    for index in range(len(chunks))
                ]

                # Store chunks and embeddings in ChromaDB.
                collection.upsert(
                    ids=chunk_ids,
                    documents=chunks,
                    embeddings=chunk_embeddings,
                )

                # Save information so it remains available after reruns.
                st.session_state.pdf_ready = True
                st.session_state.collection_name = collection_name
                st.session_state.file_name = uploaded_file.name
                st.session_state.page_count = len(pdf_reader.pages)
                st.session_state.extracted_text = extracted_text
                st.session_state.chunks = chunks

            st.success("PDF processed and stored in ChromaDB successfully.")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

    # Show document information after processing.
    if (
        st.session_state.pdf_ready
        and st.session_state.get("collection_name") == collection_name
    ):
        st.subheader("PDF information")

        st.write(f"File name: {st.session_state.file_name}")
        st.write(f"Number of pages: {st.session_state.page_count}")
        st.write(
            "Extracted characters: "
            f"{len(st.session_state.extracted_text)}"
        )
        st.write(
            f"Number of chunks created: {len(st.session_state.chunks)}"
        )

        with st.expander("View extracted text"):
            st.text_area(
                "PDF text",
                st.session_state.extracted_text,
                height=300,
            )

        with st.expander("View text chunks"):
            for index, chunk in enumerate(
                st.session_state.chunks,
                start=1,
            ):
                st.markdown(f"### Chunk {index}")
                st.write(chunk)
                st.divider()

        st.subheader("Ask a question")

        question = st.text_input(
            "Enter a question about the uploaded PDF"
        )

        if st.button("Get answer"):
            if not question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    with st.spinner("Searching the PDF and generating an answer..."):
                        # Convert the question into an embedding.
                        embedding_model = OpenAIEmbeddings(
                            model="text-embedding-3-small",
                            api_key=api_key,
                        )

                        question_embedding = (
                            embedding_model.embed_query(question)
                        )

                        # Open the collection containing the PDF chunks.
                        collection = (
                            st.session_state.chroma_client.get_collection(
                                name=st.session_state.collection_name
                            )
                        )

                        # Find the most relevant chunks.
                        number_of_results = min(
                            4,
                            len(st.session_state.chunks),
                        )

                        search_results = collection.query(
                            query_embeddings=[question_embedding],
                            n_results=number_of_results,
                            include=["documents"],
                        )

                        relevant_chunks = search_results["documents"][0]

                        context = "\n\n".join(relevant_chunks)

                        # Create the prompt sent to the LLM.
                        prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using only the context supplied below.

If the answer is not available in the context, say:
"I could not find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

                        # Send the retrieved context and question to the LLM.
                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                            api_key=api_key,
                        )

                        response = llm.invoke(prompt)

                    st.subheader("Answer")
                    st.write(response.content)

                    with st.expander("View retrieved source chunks"):
                        for index, chunk in enumerate(
                            relevant_chunks,
                            start=1,
                        ):
                            st.markdown(f"### Source {index}")
                            st.write(chunk)
                            st.divider()

                except Exception as error:
                    st.error(f"Answer generation failed: {error}")

else:
    st.info("Please upload a PDF file.")