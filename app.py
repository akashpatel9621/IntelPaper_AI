# ==========================================
# IMPORTS
# ==========================================

import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from modules.pdf_loader import load_documents
from modules.text_splitter import split_documents
from modules.embeddings import get_embedding_model
from modules.vector_store import create_vector_store
from modules.retriever import get_retriever
from modules.rag_chain import get_rag_chain
from langchain_groq import ChatGroq
from modules.suggested_questions import generate_suggested_questions



# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="IntelPaper AI",
    page_icon="📄",
    layout="wide",
)


# ==========================================
# PATHS
# ==========================================

UPLOAD_DIR = "uploaded_pdfs"
CHROMA_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
  
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = None

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []    

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

footer{
visibility:hidden;
}

.stApp{
background:#0F172A;
}

.block-container{
max-width:1150px;
padding-top:30px;
}

section[data-testid="stSidebar"]{
background:#111827;
border-right:1px solid #374151;
}

section[data-testid="stSidebar"] *{
color:white;
}

.title{
text-align:center;
margin-bottom:35px;
}

.title h1{
font-size:48px;
font-weight:700;
margin-bottom:8px;
color:white;
}

.title p{
color:#CBD5E1;
font-size:18px;
}

.upload-card{
background:#111827;
border:1px solid #374151;
border-radius:15px;
padding:24px;
margin-bottom:25px;
}

[data-testid="stFileUploader"]{
border:2px dashed #6366F1;
border-radius:12px;
padding:15px;
background:#0F172A;
}

textarea{
font-size:17px !important;
font-weight:500 !important;
}

textarea::placeholder{
font-size:16px !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📄 IntelPaper AI")

    st.markdown("---")

    st.markdown("### Navigation")

    st.write("📂 Upload PDFs")
    st.write("💬 Chat")
    st.write("💡 Suggested Questions")
    st.write("📄 Source Documents")


    st.markdown("---")

    st.info(
        """
IntelPaper AI

LangChain + ChromaDB + Groq
"""
    )


# ==========================================
# HEADER
# ==========================================

st.markdown("""

<div class="title">

<h1>📄 IntelPaper AI</h1>

<p>Understand, Search and Analyze Research Papers using AI</p>

</div>

""", unsafe_allow_html=True)


# ==========================================
# UPLOAD CARD
# ==========================================

st.markdown("""

<div class="upload-card">

<h2 style="text-align:center;color:white;">

📂 Upload Research Papers

</h2>

<p style="text-align:center;color:#CBD5E1;">

Upload one or more PDF files and start chatting with them.

</p>

</div>

""", unsafe_allow_html=True)
# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    label="Choose PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

# ==========================================
# PROCESS PDFS
# ==========================================

if uploaded_files:

    # Current uploaded file names
    uploaded_names = sorted([pdf.name for pdf in uploaded_files])

    # Process only when uploaded files change
    if st.session_state.get("loaded_files") != uploaded_names:

        with st.spinner("📄 Reading PDFs.."):

            # Reset everything
            st.session_state.messages = []
            st.session_state.rag_chain = None
            st.session_state.vector_store = None

            # Delete old upload folder
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)

            os.makedirs(UPLOAD_DIR, exist_ok=True)

            # Save uploaded PDFs
            for pdf in uploaded_files:

                pdf_path = os.path.join(
                    UPLOAD_DIR,
                    pdf.name
                )

                with open(pdf_path, "wb") as f:
                    f.write(pdf.getbuffer())

            # Load Documents
            documents = load_documents(UPLOAD_DIR)

            # Split Documents
            chunks = split_documents(documents)

            st.session_state.total_pages = len(documents)
            st.session_state.total_chunks = len(chunks)

            # Embeddings
            embedding_model = get_embedding_model()

            # Create Vector Store
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # Create Retriever
            retriever = get_retriever(vector_store)

            # Create RAG Chain
            rag_chain = get_rag_chain(retriever)

            # Save to Session
            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = rag_chain

            # Generate Suggested Questions
            st.session_state.suggested_questions = generate_suggested_questions(
                 llm,
                 documents
            )

            # Remember uploaded files
            st.session_state.loaded_files = uploaded_names

        st.success("✅ Research Papers Processed Successfully!")

# ==========================================
# SHOW UPLOADED PDFS
# ==========================================

st.markdown("## 📄 Uploaded Documents")

if st.session_state.suggested_questions:

    st.markdown("---")
    st.subheader("💡 Suggested Questions")

    for question in st.session_state.suggested_questions:
        st.markdown(f"• {question}")

if uploaded_files:

    for pdf in uploaded_files:
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📄 Pages", st.session_state.total_pages)

        with col2:
            st.metric("🧩 Chunks", st.session_state.total_chunks)

        with col3:
            st.metric("✅ Status", "Ready")

        st.markdown(
            f"""
<div style="
background:#111827;
padding:12px;
margin-bottom:10px;
border-radius:10px;
border:1px solid #374151;
color:white;
">

📄 {pdf.name}

</div>
""",
            unsafe_allow_html=True
        )

else:

    st.info("Please upload one or more research papers.")


# ==========================================
# CHAT SECTION
# ==========================================

st.markdown("---")

st.subheader("💬 Chat with your Research Papers")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask a question about your uploaded research papers..."
)


# ==========================================
# HANDLE USER QUESTION
# ==========================================

if question:

    if st.session_state.rag_chain is None:

        st.warning("⚠ Please upload and process PDF documents first.")

        st.stop()

    # -----------------------------
    # Show User Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # Assistant Response
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner("IntelPaper AI is thinking..."):

            try:

                result = st.session_state.rag_chain.invoke(
                    {
                        "input": question
                    }
                )

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

                st.markdown(answer)

                # -----------------------------
                # Show Source Documents
                # -----------------------------

                context = result.get("context", [])

                if context:

                    with st.expander("📄 Source Documents"):

                        shown_pages = set()

                        for doc in context:

                            filename = doc.metadata.get(
                                "source_file",
                                "Unknown"
                            )

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            key = (filename, page)

                            if key not in shown_pages:

                                shown_pages.add(key)

                                st.markdown(
                                    f"""
**📄 File:** {filename}

**📑 Page:** {page + 1 if isinstance(page, int) else page}
"""
                                )

                # -----------------------------
                # Save Assistant Message
                # -----------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = f"❌ Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )    
# ==========================================
# SIDEBAR UTILITIES
# ==========================================

with st.sidebar:

    st.markdown("---")

    # Clear Chat
    if st.button("🗑 Clear Chat", use_container_width=True):

        st.session_state.messages = []
        
        st.rerun()


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Uploaded PDFs",
        len(uploaded_files) if uploaded_files else 0
    )

with col2:

    st.metric(
        "Questions Asked",
        len(
            [
                m
                for m in st.session_state.messages
                if m["role"] == "user"
            ]
        )
    )

with col3:

    st.metric(
        "AI Model",
        "GPT OSS 120B"
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
padding:15px;
color:#94A3B8;
font-size:15px;'>

📄 <b>IntelPaper AI</b>

<br>

Powered by
<b>LangChain</b> •
<b>Groq</b> •
<b>HuggingFace</b> •
<b>ChromaDB</b>

<br><br>

Made for Research Paper Analysis

</div>
""",
unsafe_allow_html=True
)                