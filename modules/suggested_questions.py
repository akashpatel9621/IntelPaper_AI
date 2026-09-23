"""
suggested_questions.py
------------------------------------
Generate intelligent suggested questions
based on the uploaded document.
"""

import re
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
You are an expert Research Paper Assistant.

Read the document below and generate EXACTLY 5 useful questions
that a user can ask about this document.

Rules:
- Questions MUST be answerable from the document.
- Do NOT invent information.
- Keep questions simple.
- Do NOT repeat questions.
- Return ONLY the questions.
- Do NOT write headings or explanations.

Document:

{context}
""")


def generate_suggested_questions(llm, documents):
    """
    Generate 5 suggested questions from uploaded document.
    """

    # Select only a small number of chunks
    # to stay within Groq token limits.
    sample_docs = []

    if not documents:
        return []

    # Beginning
    sample_docs.extend(documents[:2])

    # Middle
    if len(documents) > 4:
        middle = len(documents) // 2
        sample_docs.extend(documents[middle:middle + 2])

    # End
    if len(documents) > 2:
        sample_docs.extend(documents[-2:])

    # Limit each chunk to avoid sending too many tokens
    context_parts = []

    for doc in sample_docs:
        text = doc.page_content[:1500]
        context_parts.append(text)

    context = "\n\n".join(context_parts)

    # Extra safety limit
    context = context[:8000]

    chain = PROMPT | llm

    response = chain.invoke(
        {
            "context": context
        }
    )

    text = response.content.strip()

    questions = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove numbering like 1. 2. 3)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        # Remove bullets
        line = re.sub(r"^[-•*]\s*", "", line)

        # Ignore unwanted lines
        if len(line) < 8:
            continue

        if "question" in line.lower() and ":" in line:
            continue

        questions.append(line)

    # Remove duplicates
    unique_questions = []
    seen = set()

    for q in questions:
        if q.lower() not in seen:
            unique_questions.append(q)
            seen.add(q.lower())

    return unique_questions[:5]