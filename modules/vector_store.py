"""
vector_store.py
-------------------------
Creates a fresh Chroma Vector Database.
"""

import os
import uuid
from langchain_chroma import Chroma


CHROMA_ROOT = "chroma_db"


def create_vector_store(chunks, embedding_model):
    """
    Create a new Chroma database for every upload.
    """

    # Create root folder if it doesn't exist
    os.makedirs(CHROMA_ROOT, exist_ok=True)

    # Generate unique folder
    persist_directory = os.path.join(
        CHROMA_ROOT,
        str(uuid.uuid4())
    )

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="researchmind"
    )

    return vector_store