from pathlib import Path
import requests
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Local LLM endpoint (Ollama default)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "gemma3:1b"   

IDK_PHRASE = "The information is not provided in the available documentation."

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return vector_store

def retrieve_context(query, k=4):
    """
    Retrieve relevant chunks for the query and apply a simple score-based reranking.
    """
    vs = load_vector_store()

    # Get more candidates than we finally keep
    candidates = vs.similarity_search_with_score(query, k=k)

    # Sort by score (lower score = closer match)
    candidates_sorted = sorted(candidates, key=lambda pair: pair[1])

    context_text = ""
    sources = set()

    # Take the top N after reranking (you can tune this)
    top_n = 3
    for doc, score in candidates_sorted[:top_n]:
        #print("Using chunk with score:", score)
        context_text += doc.page_content + "\n\n"
        if "source" in doc.metadata:
            sources.add(Path(doc.metadata["source"]).name)

    return context_text, list(sources)


def call_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"LLM call failed: {response.text}")
    return response.json().get("response", "").strip()

def build_prompt(question, context):
    return f"""
You are an internal policy assistant. 
You MUST answer ONLY using the context provided below.

First, read the context carefully. If the context clearly contains information that answers the question, use it to give a concise answer in a full sentence.
Only if the context does not contain relevant information, reply strictly with:
"{IDK_PHRASE}"


Context:
--------------------
{context}
--------------------

Question: {question}

Answer:
"""

def ask(question):
    context, sources = retrieve_context(question)

    prompt = build_prompt(question, context)
    answer = call_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    result = ask("How many days per week can I work remotely?")
    print(result)

