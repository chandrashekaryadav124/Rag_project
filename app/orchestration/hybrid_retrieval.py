
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Any

from app.core.security import UserPrincipal
from app.orchestration.role_retrieval import retrieve_role_data
from app.orchestration.document_retrieval import (
    retrieve_authorized_documents,
)


def retrieve_hybrid_context(
    principal: UserPrincipal,
    question: str,
    task: Optional[str] = None,
    order_id: Optional[str] = None,
    limit: int = 10,
    document_retriever: Optional[Callable[..., Any]] = None,
) -> dict:
    """
    Combines authorized SQL retrieval with role-authorized
    document retrieval.
    """

    if not principal.is_active:
        raise PermissionError("Inactive users cannot retrieve information.")

    if not question or not question.strip():
        raise ValueError("Question is required.")

    cleaned_question = question.strip()
    retriever = (
        document_retriever
        if document_retriever is not None
        else retrieve_authorized_documents
    )

    structured_data = None

    # Run SQL and document retrieval concurrently when both are needed.
    if task:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sql_future = executor.submit(
                retrieve_role_data,
                principal=principal,
                task=task,
                order_id=order_id,
                limit=limit,
            )
            doc_future = executor.submit(
                retriever,
                question=cleaned_question,
                principal=principal,
            )
            structured_data = sql_future.result()
            knowledge_documents = doc_future.result()
    else:
        knowledge_documents = retriever(
            question=cleaned_question,
            principal=principal,
        )

    # Return both retrieval sources separately
    return {
        "user_id": principal.user_id,
        "role": principal.role.value,
        "question": cleaned_question,
        "structured_data": structured_data,
        "knowledge_documents": knowledge_documents,
        "has_structured_data": bool(structured_data),
        "has_knowledge_documents": bool(knowledge_documents),
    }
