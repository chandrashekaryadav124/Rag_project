
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


@patch(
    "app.orchestration.document_retrieval.retrieve_supabase_documents"
)
def test_customer_can_retrieve_documents(mock_retrieve):
    mock_retrieve.return_value = ["customer document"]

    result = retrieve_authorized_documents(
        "return policy",
        make_user(UserRole.CUSTOMER)
    )

    assert result == ["customer document"]

    mock_retrieve.assert_called_once_with(
        question="return policy",
        role="CUSTOMER",
        top_k=4
    )


@pytest.mark.parametrize(
    "user_role, expected_filter_role",
    [
        (UserRole.CUSTOMER, "CUSTOMER"),
        (UserRole.SELLER, "SELLER"),
        (UserRole.SUPPORT, "SUPPORT"),
        (UserRole.OPERATIONS, "OPERATIONS"),
        (UserRole.ADMIN, "ADMIN"),
    ],
)
@patch(
    "app.orchestration.document_retrieval.retrieve_supabase_documents"
)
def test_all_five_roles_retrieve_only_their_own_role_documents(
    mock_retrieve,
    user_role,
    expected_filter_role,
):
    mock_retrieve.return_value = [{"content": f"{expected_filter_role} doc"}]

    result = retrieve_authorized_documents(
        "policy question",
        make_user(user_role),
    )

    assert result == [{"content": f"{expected_filter_role} doc"}]
    mock_retrieve.assert_called_once_with(
        question="policy question",
        role=expected_filter_role,
        top_k=4,
    )


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
