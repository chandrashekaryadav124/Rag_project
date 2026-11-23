
# ==========================================
# CUSTOMER RAG - DOCUMENT RETRIEVAL
# ==========================================

import os

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_FOLDER = os.path.join(BASE_DIR, "vectorstore")

COLLECTION_NAME = "customer_knowledge_base"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 4


# ==========================================
# STEP 1: LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def create_embeddings():

    print("Loading customer embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


# ==========================================
# STEP 2: LOAD CHROMADB
# ==========================================

@lru_cache(maxsize=1)
def load_vector_database():

    if not os.path.exists(VECTOR_FOLDER):

        raise FileNotFoundError(
            "Vector database not found. Run ingestion.py first."
        )

    embeddings = create_embeddings()

    database = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_FOLDER
    )

    count = database._collection.count()

    if count == 0:

        raise ValueError(
            "Customer knowledge base is empty."
        )

    print(f"Customer knowledge base loaded: {count} chunks")

    return database


# ==========================================
# STEP 3: RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(question: str, top_k: int = DEFAULT_TOP_K):

    question = question.strip()

    if not question:
        return []

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    database = load_vector_database()

    results = database.similarity_search(
        question,
        k=top_k,
        filter={"role": "customer"}
    )

    return results


# ==========================================
# STEP 4: DATABASE INFORMATION
# ==========================================

def get_database_info():

    database = load_vector_database()

    return {
        "collection": COLLECTION_NAME,
        "total_chunks": database._collection.count(),
        "embedding_model": EMBEDDING_MODEL
    }


# ==========================================
# STEP 5: TERMINAL TEST
# ==========================================

if __name__ == "__main__":

    print("\nCUSTOMER RAG RETRIEVAL TEST")

    print(get_database_info())

    while True:

        question = input("\nEnter question (exit to stop): ")

        if question.lower().strip() == "exit":
            break

        try:

            documents = retrieve_documents(question)

            if not documents:

                print("No relevant documents found.")

                continue

            for index, doc in enumerate(documents, start=1):

                print(f"\nResult {index}")

                print("Source:", doc.metadata.get("source_file"))

                print("Page:", doc.metadata.get("page"))

                print("Content:", doc.page_content[:400])

        except Exception as error:

            print("Retrieval Error:", str(error))
