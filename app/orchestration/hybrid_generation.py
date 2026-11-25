
import json
import time
from typing import Optional

from app.core.prompts import build_hybrid_prompt_value
from app.core.security import UserPrincipal
from app.orchestration.hybrid_retrieval import retrieve_hybrid_context
from app.orchestration.conversation_memory import (
    get_or_create_session,
    get_recent_messages,
    save_message,
)
from app.generation import create_llm, extract_answer


def prepare_document_context(documents):
    context_parts = []
    sources = []

    for index, document in enumerate(documents, start=1):
        metadata = getattr(document, "metadata", {}) or {}

        if isinstance(document, dict):
            doc_metadata = document.get("metadata") or {}
            content = document.get("content", "")
            source_name = (
                document.get("source")
                or doc_metadata.get("source_file")
                or "Knowledge Base"
            )
            page_number = document.get("page", doc_metadata.get("page"))
        else:
            content = getattr(document, "page_content", "")
            source_name = metadata.get(
                "source_file",
                metadata.get("source", "Knowledge Base")
            )
            page_number = metadata.get("page")

        context_parts.append(
            f"""
DOCUMENT SOURCE {index}
Source: {source_name}
Page: {page_number}

Content:
{content}
"""
        )

        sources.append({
            "type": "document",
            "source": source_name,
            "page": page_number,
        })

    return "\n".join(context_parts), sources


def prepare_structured_context(structured_data):
    if not structured_data:
        return "", []

    context = json.dumps(
        structured_data,
        indent=2,
        default=str
    )

    sources = [{
        "type": "structured_data",
        "source": "Authorized application database",
    }]

    return context, sources


def build_hybrid_prompt(
    role: str,
    question: str,
    structured_context: str,
    document_context: str,
    conversation_history: str = "",
):
    return build_hybrid_prompt_value(
        role=role,
        question=question,
        structured_context=structured_context,
        document_context=document_context,
        conversation_history=conversation_history,
    )


def generate_hybrid_answer(
    principal: UserPrincipal,
    question: str,
    task: Optional[str] = None,
    order_id: Optional[str] = None,
    limit: int = 10,
    session_id: Optional[str] = None,
) -> dict:

    question = question.strip() if question else ""

    if not question:
        raise ValueError("Question cannot be empty.")

    if not principal.is_active:
        raise PermissionError("Inactive users cannot ask questions.")

    # 1. Create or validate the conversation session.
    session = get_or_create_session(
        principal=principal,
        session_id=session_id,
    )
    active_session_id = session["id"]

    # 2. Load previous messages BEFORE saving the current question.
    history_messages = get_recent_messages(
        principal=principal,
        session_id=active_session_id,
        limit=10,
    )

    conversation_history = "\n".join(
        f'{message["message_type"].upper()}: {message["content"]}'
        for message in history_messages
    )

    # 3. Retrieve authorized SQL and document context.
    retrieval_start = time.perf_counter()

    hybrid_context = retrieve_hybrid_context(
        principal=principal,
        question=question,
        task=task,
        order_id=order_id,
        limit=limit,
    )

    retrieval_time = round(
        time.perf_counter() - retrieval_start,
        2
    )

    structured_data = hybrid_context["structured_data"]
    documents = hybrid_context["knowledge_documents"]

    print(
        f"[HYBRID DEBUG] role={principal.role.value}, "
        f"task={task}, "
        f"structured_order_count={len(structured_data or [])}"
    )

    # 4. Prepare context.
    structured_context, sql_sources = prepare_structured_context(
        structured_data
    )

    document_context, document_sources = prepare_document_context(
        documents
    )

    sources = sql_sources + document_sources

    found_context = bool(
        structured_context.strip() or document_context.strip()
    )

    # 5. Handle missing context and persist user + assistant turn.
    if not found_context:
        answer = (
            "I could not find sufficient authorized information "
            "to answer your question. Please contact the "
            "appropriate support team for assistance."
        )

        save_message(
            principal=principal,
            session_id=active_session_id,
            message_type="user",
            content=question,
        )
        save_message(
            principal=principal,
            session_id=active_session_id,
            message_type="assistant",
            content=answer,
        )

        return {
            "answer": answer,
            "role": principal.role.value,
            "session_id": active_session_id,
            "sources": [],
            "found_context": False,
            "retrieval_time": retrieval_time,
            "generation_time": 0,
        }

    # 6. Build prompt with conversation history.
    prompt = build_hybrid_prompt(
        role=principal.role.value,
        question=question,
        structured_context=structured_context,
        document_context=document_context,
        conversation_history=conversation_history,
    )

    # 7. Generate answer using Groq.
    model = create_llm()

    generation_start = time.perf_counter()
    response = model.invoke(prompt)

    generation_time = round(
        time.perf_counter() - generation_start,
        2
    )

    answer = extract_answer(response)

    # 8. Persist completed user + assistant turn (avoiding orphaned user messages on failure).
    save_message(
        principal=principal,
        session_id=active_session_id,
        message_type="user",
        content=question,
    )
    save_message(
        principal=principal,
        session_id=active_session_id,
        message_type="assistant",
        content=answer,
    )

    # 9. Return answer and session ID.
    return {
        "answer": answer,
        "role": principal.role.value,
        "session_id": active_session_id,
        "sources": sources,
        "found_context": True,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
    }
