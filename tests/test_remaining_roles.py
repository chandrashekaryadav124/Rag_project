
import pytest

from app.core.security import UserPrincipal, UserRole

from app.authorized_support_sql import (
    get_support_order_summary,
    get_support_tickets,
)

from app.authorized_operations_sql import (
    get_shipment_summary,
    get_inventory_snapshot_summary,
)

from app.authorized_admin_sql import get_system_analytics


def make_user(role):
    return UserPrincipal(
        user_id=f"TEST-{role.value.upper()}",
        role=role,
        is_active=True,
    )


# SUPPORT TESTS

def test_customer_cannot_access_support_tickets():
    with pytest.raises(PermissionError):
        get_support_tickets(make_user(UserRole.CUSTOMER))


def test_seller_cannot_access_support_order_summary():
    with pytest.raises(PermissionError):
        get_support_order_summary(
            make_user(UserRole.SELLER),
            "ORDER-TEST-01",
        )


def test_support_rejects_empty_order_id():
    with pytest.raises(ValueError):
        get_support_order_summary(
            make_user(UserRole.SUPPORT),
            "",
        )


# OPERATIONS TESTS

def test_customer_cannot_access_shipment_summary():
    with pytest.raises(PermissionError):
        get_shipment_summary(
            make_user(UserRole.CUSTOMER),
            "ORDER-TEST-01",
        )


def test_support_cannot_access_inventory_overview():
    with pytest.raises(PermissionError):
        get_inventory_snapshot_summary(
            make_user(UserRole.SUPPORT),
        )


def test_operations_rejects_empty_order_id():
    with pytest.raises(ValueError):
        get_shipment_summary(
            make_user(UserRole.OPERATIONS),
            "",
        )


# ADMIN TESTS

def test_customer_cannot_access_system_analytics():
    with pytest.raises(PermissionError):
        get_system_analytics(make_user(UserRole.CUSTOMER))


def test_seller_cannot_access_system_analytics():
    with pytest.raises(PermissionError):
        get_system_analytics(make_user(UserRole.SELLER))


def test_support_cannot_access_system_analytics():
    with pytest.raises(PermissionError):
        get_system_analytics(make_user(UserRole.SUPPORT))


def test_operations_cannot_access_system_analytics():
    with pytest.raises(PermissionError):
        get_system_analytics(make_user(UserRole.OPERATIONS))
