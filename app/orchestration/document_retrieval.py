
from app.core.security import UserPrincipal
from app.supabase_retrieval import retrieve_supabase_documents


def retrieve_authorized_documents(
    question: str,
    principal: UserPrincipal,
    top_k: int = 4
):
    """
    Retrieve approved documents from Supabase
    for the authenticated user's role.
    """

    if not principal.is_active:
        raise PermissionError("Inactive users cannot retrieve documents.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    role = principal.role.value.upper()

    return retrieve_supabase_documents(
        question=question,
        role=role,
        top_k=top_k,
    )
