import os
import shutil
import uuid

import streamlit as st
import pandas as pd
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Mr. Analyst",
    layout="wide"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================================================
# INITIALIZE MODELS
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

client = chromadb.PersistentClient(
    path="./chroma_db"
)

COLLECTION_NAME = "rag_documents"

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def reset_collection():

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def get_collection():

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def read_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_csv(uploaded_file):

    df = pd.read_csv(uploaded_file)

    chunks = []

    for _, row in df.iterrows():

        row_text = " | ".join(
            [
                f"{col}: {row[col]}"
                for col in df.columns
            ]
        )

        chunks.append(row_text)

    return chunks


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_text(text)


def process_files(files):

    collection = reset_collection()

    all_chunks = []

    progress = st.progress(0)

    total_files = len(files)

    for idx, file in enumerate(files):

        file_name = file.name.lower()

        if file_name.endswith(".pdf"):

            text = read_pdf(file)

            chunks = chunk_text(text)

            all_chunks.extend(chunks)

        elif file_name.endswith(".csv"):

            chunks = read_csv(file)

            all_chunks.extend(chunks)

        progress.progress(
            (idx + 1) / total_files
        )

    if len(all_chunks) == 0:
        return 0

    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True
    ).tolist()

    ids = [
        str(uuid.uuid4())
        for _ in range(len(all_chunks))
    ]

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings
    )

    return len(all_chunks)


def retrieve_context(question, top_k=5):

    collection = get_collection()

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]

    return docs


def ask_groq(question, context_chunks):

    client_groq = Groq(
        api_key=GROQ_API_KEY
    )

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful analyst.

Answer only using the provided context.

Context:
{context}

Question:
{question}
"""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# =====================================================
# SESSION STATE
# =====================================================

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# =====================================================
# UI
# =====================================================

st.title("Mr. Analyst")

st.markdown("---")

# =====================================================
# DOCUMENT UPLOAD SECTION
# =====================================================

st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs and CSVs",
    type=["pdf", "csv"],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)

with col1:

    if st.button("Process Documents"):

        if not uploaded_files:

            st.warning(
                "Please upload files first."
            )

        else:

            with st.spinner(
                "Reading, chunking and indexing..."
            ):

                count = process_files(
                    uploaded_files
                )

                st.session_state.indexed = True
                st.session_state.chunk_count = count

            st.success(
                f"{count} chunks indexed successfully."
            )

with col2:

    if st.button("Reset Knowledge Base"):

        try:
            client.delete_collection(
                COLLECTION_NAME
            )
        except:
            pass

        st.session_state.indexed = False
        st.session_state.chunk_count = 0

        st.success("Knowledge base cleared.")

# =====================================================
# STATUS
# =====================================================

st.markdown("---")

if st.session_state.indexed:

    st.success(
        f"Knowledge Base Ready | Chunks Indexed: {st.session_state.chunk_count}"
    )

else:

    st.info(
        "No documents indexed yet."
    )

# =====================================================
# QUESTION SECTION
# =====================================================

st.markdown("---")

st.subheader("Ask Questions")

query = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if not st.session_state.indexed:

        st.error(
            "Please process documents first."
        )

    elif not query.strip():

        st.error(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Retrieving context..."
        ):

            context_docs = retrieve_context(
                query,
                top_k=5
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = ask_groq(
                query,
                context_docs
            )

        st.markdown("## Answer")

        st.write(answer)

        with st.expander(
            "Retrieved Context"
        ):

            for i, doc in enumerate(
                context_docs,
                start=1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(doc)

                st.markdown("---")
