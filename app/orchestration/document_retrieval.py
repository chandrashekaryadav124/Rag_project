
from app.core.security import UserPrincipal
from app.retrieval import retrieve_documents


def retrieve_authorized_documents(
    question: str,
    principal: UserPrincipal,
    top_k: int = 4
):
    """
    Retrieve documents only from a knowledge base
    authorized for the authenticated user's role.

    Currently, only the Customer knowledge base exists.
    Other roles can still use authorized SQL retrieval.
    """

    if not principal.is_active:
        raise PermissionError("Inactive users cannot retrieve documents.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    role = principal.role.value

    if role == "customer":
        return retrieve_documents(
            question=question,
            top_k=top_k
        )

    # Other role-specific document collections
    # will be connected after their knowledge bases are created.
    return []
