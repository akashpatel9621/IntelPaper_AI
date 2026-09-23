"""
embeddings.py
-------------------------
Creates the HuggingFace Embedding Model.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Returns the embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model