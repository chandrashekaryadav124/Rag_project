
import pytest
from unittest.mock import patch
from app.core.security import UserPrincipal, UserRole
from app.orchestration.hybrid_retrieval import retrieve_hybrid_context


def make_user(role):
    return UserPrincipal(
        user_id="TEST-USER",
        role=role,
        is_active=True,
    )


def test_hybrid_retrieval_rejects_empty_question():
    with pytest.raises(ValueError):
        retrieve_hybrid_context(
            make_user(UserRole.CUSTOMER),
            "",
        )



@patch(
    "app.orchestration.hybrid_retrieval.retrieve_authorized_documents",
    return_value=[]
)
def test_hybrid_retrieval_without_sources(mock_retriever):
    result = retrieve_hybrid_context(
        make_user(UserRole.CUSTOMER),
        "How can I return a product?",
    )

    assert result["structured_data"] is None
    assert result["knowledge_documents"] == []
    assert result["has_structured_data"] is False
    assert result["has_knowledge_documents"] is False

    mock_retriever.assert_called_once()


def test_hybrid_retrieval_uses_document_callback():
    def fake_retriever(question, principal):
        return [
            {
                "content": "Returns can be requested within the policy period.",
                "source": "customer_policy.pdf",
            }
        ]

    result = retrieve_hybrid_context(
        make_user(UserRole.CUSTOMER),
        "What is the return policy?",
        document_retriever=fake_retriever,
    )

    assert result["has_knowledge_documents"] is True
    assert len(result["knowledge_documents"]) == 1


def test_hybrid_retrieval_rejects_inactive_user():
    with pytest.raises(PermissionError):
        retrieve_hybrid_context(
            UserPrincipal(
                user_id="INACTIVE",
                role=UserRole.CUSTOMER,
                is_active=False,
            ),
            "Where is my order?",
        )
