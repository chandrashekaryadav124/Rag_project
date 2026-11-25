
from app.core.supabase_client import get_supabase_client
from app.retrieval import create_embeddings


DEFAULT_TOP_K = 4


def retrieve_supabase_documents(
    question: str,
    role: str = "CUSTOMER",
    top_k: int = DEFAULT_TOP_K
):
    question = question.strip()

    if not question:
        return []

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    allowed_roles = {
        "CUSTOMER",
        "SELLER",
        "SUPPORT",
        "OPERATIONS",
        "ADMIN"
    }

    if role not in allowed_roles:
        raise ValueError("Invalid RAG role")

    embeddings = create_embeddings()
    query_embedding = embeddings.embed_query(question)

    supabase = get_supabase_client()

    result = supabase.rpc(
        "match_knowledge_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filter_role": role
        }
    ).execute()

    return result.data or []


if __name__ == "__main__":
    results = retrieve_supabase_documents(
        "What is the return policy?",
        role="CUSTOMER",
        top_k=4
    )

    print("\nSUPABASE SEMANTIC SEARCH RESULTS")

    for index, item in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Title:", item["title"])
        print("Source:", item["source"])
        print("Similarity:", round(item["similarity"], 4))
        print("Content:", item["content"][:400])
