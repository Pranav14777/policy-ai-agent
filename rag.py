# rag.py
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).parent
POLICY_DIR = BASE_DIR / "policies"
CHROMA_DIR = BASE_DIR / "chroma_db"

def load_policy_documents():
    """
    Load all .md files from the policies directory as LangChain Documents.
    """
    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    # Load all markdown files
    loader = DirectoryLoader(
        str(POLICY_DIR),
        glob="*.md",
        loader_cls=TextLoader,  # load as plain text
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} policy documents.")
    return docs


def split_documents(docs):
    """
    Split documents into smaller chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks):
    """
    Create or update a Chroma vector store from document chunks.
    """
    # Use a small sentence-transformers model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.persist()
    print(f"Vector store created at: {CHROMA_DIR}")
    return vector_store

def test_retrieval(vector_store):
    """
    Simple test to see what chunks are retrieved for a sample question.
    """
    query = "When does the probation evaluation take place for new employees?"
    results = vector_store.similarity_search(query, k=3)
    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
        print()


if __name__ == "__main__":
    docs = load_policy_documents()
    chunks = split_documents(docs)
    vs = build_vector_store(chunks)
    test_retrieval(vs)

