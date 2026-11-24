
from unittest.mock import patch

import pytest

from app.core.security import UserPrincipal, UserRole
from app.orchestration.document_retrieval import (
    retrieve_authorized_documents
)


def make_user(role, active=True):
    return UserPrincipal(
        user_id="test-user",
        role=role,
        is_active=active
    )


@patch("app.orchestration.document_retrieval.retrieve_documents")
def test_customer_can_retrieve_documents(mock_retrieve):
    mock_retrieve.return_value = ["customer document"]

    result = retrieve_authorized_documents(
        "return policy",
        make_user(UserRole.CUSTOMER)
    )

    assert result == ["customer document"]
    mock_retrieve.assert_called_once()


@patch("app.orchestration.document_retrieval.retrieve_documents")
def test_seller_cannot_access_customer_documents(mock_retrieve):
    result = retrieve_authorized_documents(
        "return policy",
        make_user(UserRole.SELLER)
    )

    assert result == []
    mock_retrieve.assert_not_called()


@patch("app.orchestration.document_retrieval.retrieve_documents")
def test_support_cannot_access_customer_documents(mock_retrieve):
    result = retrieve_authorized_documents(
        "return policy",
        make_user(UserRole.SUPPORT)
    )

    assert result == []
    mock_retrieve.assert_not_called()


def test_inactive_user_is_rejected():
    with pytest.raises(PermissionError):
        retrieve_authorized_documents(
            "return policy",
            make_user(UserRole.CUSTOMER, active=False)
        )


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        retrieve_authorized_documents(
            "",
            make_user(UserRole.CUSTOMER)
        )
