
# ==========================================
# CUSTOMER RAG - DOCUMENT INGESTION
# ==========================================

import os
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_FOLDER = os.path.join(BASE_DIR, "data", "customers_KB")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vectorstore")

COLLECTION_NAME = "customer_knowledge_base"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# STEP 1: LOAD PDF DOCUMENTS
# ==========================================

def load_pdf():

    documents = []

    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_FOLDER}"
        )

    for file_name in sorted(os.listdir(PDF_FOLDER)):

        if file_name.lower().endswith(".pdf"):

            file_path = os.path.join(PDF_FOLDER, file_name)

            loader = PyPDFLoader(file_path)

            pdf_documents = loader.load()

            for doc in pdf_documents:

                doc.metadata["role"] = "customer"

                doc.metadata["source_file"] = file_name

                if file_name.startswith("02_"):
                    doc.metadata["policy_status"] = (
                        "approved_project_baseline"
                    )
                else:
                    doc.metadata["policy_status"] = "reference"

            documents.extend(pdf_documents)

            print(
                f"Loaded: {file_name} "
                f"({len(pdf_documents)} pages)"
            )

    if not documents:
        raise ValueError("No PDF documents found.")

    print("Total PDF pages:", len(documents))

    return documents


# ==========================================
# STEP 2: SPLIT DOCUMENTS
# ==========================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print("Total chunks:", len(chunks))

    return chunks


# ==========================================
# STEP 3: CREATE EMBEDDINGS
# ==========================================

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ==========================================
# STEP 4: STORE IN CHROMADB
# ==========================================

def store_in_chromadb(chunks, embeddings):

    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    client = chromadb.PersistentClient(
        path=VECTOR_FOLDER
    )

    # Delete only the customer collection
    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Previous customer collection deleted.")

    except Exception:

        print("No previous customer collection found.")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_FOLDER
    )

    print("Customer documents stored successfully.")

    print(
        "Total stored chunks:",
        vector_db._collection.count()
    )

    return vector_db


# ==========================================
# STEP 5: COMPLETE INGESTION PIPELINE
# ==========================================

def main():

    print("\n================================")
    print("CUSTOMER RAG INGESTION STARTED")
    print("================================\n")

    try:

        documents = load_pdf()

        chunks = split_documents(documents)

        embeddings = create_embeddings()

        store_in_chromadb(chunks, embeddings)

        print("\nCUSTOMER RAG INGESTION COMPLETED!")

    except Exception as error:

        print("\nINGESTION ERROR:", str(error))

        raise


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
