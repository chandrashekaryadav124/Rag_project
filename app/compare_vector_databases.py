
from app.retrieval import retrieve_documents, create_embeddings
from app.supabase_retrieval import retrieve_supabase_documents

QUESTIONS = [
    "What is the return policy?",
    "How many days do customers have to request a return?",
    "Who pays return shipping for damaged products?",
    "How can a customer request a return?",
    "When will customers receive their refund?"
]


def main():
    print("\n========== CHROMADB VS SUPABASE ==========")

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        # ChromaDB retrieval
        chroma_results = retrieve_documents(question, top_k=4)

        # Supabase retrieval
        supabase_results = retrieve_supabase_documents(
            question,
            role="CUSTOMER",
            top_k=4
        )

        print("\n--- CHROMADB RESULTS ---")

        for i, doc in enumerate(chroma_results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source_file')}")
            print("Content:", doc.page_content[:250].replace("\n", " "))

        print("\n--- SUPABASE RESULTS ---")

        for i, doc in enumerate(supabase_results, 1):
            print(f"\n{i}. Source: {doc.get('source')}")
            print("Content:", doc.get('content', '')[:250].replace("\n", " "))

        print("\nResult counts:")
        print("ChromaDB:", len(chroma_results))
        print("Supabase:", len(supabase_results))

    print("\n========== COMPARISON COMPLETED ==========")


if __name__ == "__main__":
    main()
