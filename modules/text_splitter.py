"""
text_splitter.py
-------------------------
Splits loaded PDF documents into chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into chunks.

    Parameters
    ----------
    documents : list
        List of LangChain Documents.

    Returns
    -------
    list
        List of chunked Documents.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks