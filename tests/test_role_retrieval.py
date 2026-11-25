
import pytest

from app.core.security import UserPrincipal, UserRole
from app.orchestration.role_retrieval import retrieve_role_data


def make_user(role):
    return UserPrincipal(
        user_id=f"TEST-{role.value.upper()}",
        role=role,
        is_active=True,
    )


def test_customer_cannot_request_seller_products():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.CUSTOMER),
            "own_products",
        )


def test_seller_cannot_request_customer_orders():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.SELLER),
            "own_orders",
        )


def test_support_cannot_request_system_analytics():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.SUPPORT),
            "system_analytics",
        )


def test_operations_cannot_request_support_tickets():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.OPERATIONS),
            "support_tickets",
        )


def test_customer_cannot_request_admin_analytics():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.CUSTOMER),
            "system_analytics",
        )


def test_invalid_task_is_rejected():
    with pytest.raises(PermissionError):
        retrieve_role_data(
            make_user(UserRole.ADMIN),
            "execute_custom_sql",
        )


def test_task_must_be_string():
    with pytest.raises(ValueError):
        retrieve_role_data(
            make_user(UserRole.ADMIN),
            None,
        )


def test_support_order_and_tickets_returns_combined_data(monkeypatch):
    monkeypatch.setattr(
        "app.orchestration.role_retrieval.get_support_order_summary",
        lambda principal, order_id: {"order_id": order_id, "order_status": "Processing"},
    )
    monkeypatch.setattr(
        "app.orchestration.role_retrieval.get_support_tickets_by_order",
        lambda principal, order_id: [{"ticket_id": "TCK-1", "order_id": order_id}],
    )

    result = retrieve_role_data(
        make_user(UserRole.SUPPORT),
        "order_and_tickets",
        order_id="ORD-123",
    )

    assert result["order"]["order_id"] == "ORD-123"
    assert len(result["tickets"]) == 1
