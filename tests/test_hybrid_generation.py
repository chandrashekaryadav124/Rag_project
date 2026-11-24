
from unittest.mock import patch, MagicMock

import pytest

from app.core.security import UserPrincipal, UserRole
from app.orchestration.hybrid_generation import (
    generate_hybrid_answer,
)


def make_customer():
    return UserPrincipal(
        user_id="TEST-CUSTOMER",
        role=UserRole.CUSTOMER,
        is_active=True,
    )


@patch("app.orchestration.hybrid_generation.create_llm")
@patch("app.orchestration.hybrid_generation.retrieve_hybrid_context")
def test_hybrid_generation_with_documents(
    mock_retrieval,
    mock_create_llm,
):
    mock_retrieval.return_value = {
        "structured_data": None,
        "knowledge_documents": [
            {
                "content": "Returns can be requested within 10 days.",
                "source": "customer_policy.pdf",
                "page": 2,
            }
        ],
    }

    mock_model = MagicMock()
    mock_model.invoke.return_value.content = (
        "You can request a return within 10 days."
    )
    mock_create_llm.return_value = mock_model

    result = generate_hybrid_answer(
        principal=make_customer(),
        question="What is the return period?",
    )

    assert result["found_context"] is True
    assert "10 days" in result["answer"]
    assert len(result["sources"]) == 1
    mock_model.invoke.assert_called_once()


@patch("app.orchestration.hybrid_generation.create_llm")
@patch("app.orchestration.hybrid_generation.retrieve_hybrid_context")
def test_hybrid_generation_without_context(
    mock_retrieval,
    mock_create_llm,
):
    mock_retrieval.return_value = {
        "structured_data": None,
        "knowledge_documents": [],
    }

    result = generate_hybrid_answer(
        principal=make_customer(),
        question="Unknown question",
    )

    assert result["found_context"] is False
    assert result["sources"] == []
    assert result["generation_time"] == 0
    mock_create_llm.assert_not_called()


def test_hybrid_generation_rejects_empty_question():
    with pytest.raises(ValueError):
        generate_hybrid_answer(
            principal=make_customer(),
            question="",
        )



def test_hybrid_generation_rejects_inactive_user():
    with pytest.raises(PermissionError, match="User account is inactive"):
        UserPrincipal(
            user_id="INACTIVE",
            role=UserRole.CUSTOMER,
            is_active=False,
        )
