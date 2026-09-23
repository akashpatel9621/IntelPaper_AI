"""
rag_chain.py
-----------------------------------
Creates the Retrieval-Augmented Generation (RAG) Chain.
"""

import os

from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq

from modules.prompt import get_prompt


# Load Environment Variables
load_dotenv()


def get_rag_chain(retriever):
    """
    Create and return the RAG Chain.

    Parameters
    ----------
    retriever : VectorStoreRetriever

    Returns
    -------
    Runnable
        LangChain Retrieval Chain
    """

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # Prompt
    prompt = get_prompt()

    # Document Chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # Retrieval Chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain