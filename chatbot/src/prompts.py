 
from langchain.schema import Document
from typing import List

SYSTEM_INSTRUCTIONS = """
You are a helpful assistant answering user queries using only the provided context (documents).
If the document does not contain the answer, say you don't know and optionally provide a short suggestion.
Be concise and show the source for any direct facts (metadata.source).
"""

def make_prompt_for_answer(user_query: str, docs: List[Document]) -> str:
    context_texts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", f"doc_{i}")
        snippet = d.page_content.strip().replace("\n", " ")
        # optionally trim large snippet
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        context_texts.append(f"=== SOURCE {i} ({src}) ===\n{snippet}\n")

    context_block = "\n\n".join(context_texts)
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        "INSTRUCTIONS: Answer the question based only on the context above. "
        "If you must guess, say 'I am unsure' and explain what additional info you'd need."
    )
    return prompt
