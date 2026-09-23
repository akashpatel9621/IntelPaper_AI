"""
retriever.py
-------------------------
Creates Retriever from Chroma Vector Store.
"""

from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(vector_store) -> VectorStoreRetriever:
    """
    Create a similarity-based retriever.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 12
        }
    )

    return retriever