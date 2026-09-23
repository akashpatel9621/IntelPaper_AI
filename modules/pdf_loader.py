from langchain_community.document_loaders import PyMuPDFLoader
import os


def load_documents(pdf_directory):
    """
    Load all PDFs from the uploaded_pdfs folder.
    Returns a list of LangChain Documents.
    """

    documents = []

    pdf_files = sorted(
        [
            file
            for file in os.listdir(pdf_directory)
            if file.lower().endswith(".pdf")
        ]
    )

    for pdf in pdf_files:

        pdf_path = os.path.join(pdf_directory, pdf)

        loader = PyMuPDFLoader(pdf_path)

        docs = loader.load()

        # Store filename in metadata
        for doc in docs:
            doc.metadata["source_file"] = pdf

        documents.extend(docs)

    return documents