
import pytest


from app.core.security import UserPrincipal, UserRole, require_roles
from app.core.role_policies import has_permission, get_role_permissions

from app.core.security import (
    UserPrincipal,
    UserRole,
    require_roles,
)
from app.core.role_policies import (
    has_permission,
    get_role_permissions,
)


def test_customer_principal_creation():
    user = UserPrincipal(
        user_id="USR-TEST-01",
        role=UserRole.CUSTOMER,
        is_active=True,
    )

    assert user.user_id == "USR-TEST-01"
    assert user.role == UserRole.CUSTOMER


def test_inactive_user_rejected():
    with pytest.raises(PermissionError):
        UserPrincipal(
            user_id="USR-TEST-02",
            role=UserRole.CUSTOMER,
            is_active=False,
        )


def test_invalid_role_rejected():
    with pytest.raises(ValueError):
        UserPrincipal(
            user_id="USR-TEST-03",
            role="unknown",
            is_active=True,
        )


def test_customer_role_authorization():
    user = UserPrincipal(
        user_id="USR-CUSTOMER",
        role=UserRole.CUSTOMER,
        is_active=True,
    )

    assert require_roles(user, {UserRole.CUSTOMER}) is True

    with pytest.raises(PermissionError):
        require_roles(user, {UserRole.ADMIN})


def test_all_five_roles_have_permission_entries():
    expected_roles = {
        UserRole.CUSTOMER,
        UserRole.SELLER,
        UserRole.SUPPORT,
        UserRole.OPERATIONS,
        UserRole.ADMIN,
    }

    assert set(UserRole) == expected_roles

    for role in expected_roles:
        assert isinstance(get_role_permissions(role), list)


def test_admin_has_system_management_permission():
    assert has_permission(
        UserRole.ADMIN,
        "system_management"
    )


def test_customer_cannot_manage_system():
    assert not has_permission(
        UserRole.CUSTOMER,
        "system_management"
    )


def test_seller_cannot_access_customer_order_lookup():
    assert not has_permission(
        UserRole.SELLER,
        "order_lookup"
    )


def test_customer_can_access_own_orders():
    assert has_permission(
        UserRole.CUSTOMER,
        "own_orders"
    )


@pytest.mark.parametrize(
    "role, permission, expected",
    [
        (UserRole.CUSTOMER, "own_orders", True),
        (UserRole.CUSTOMER, "system_management", False),

        (UserRole.SELLER, "own_products", True),
        (UserRole.SELLER, "order_lookup", False),

        (UserRole.SUPPORT, "customer_support", True),
        (UserRole.SUPPORT, "system_management", False),

        (UserRole.OPERATIONS, "inventory_overview", True),
        (UserRole.OPERATIONS, "system_management", False),

        (UserRole.ADMIN, "system_management", True),
        (UserRole.ADMIN, "system_analytics", True),
    ],
)
def test_all_role_permissions(role, permission, expected):
    assert has_permission(role, permission) is expected


@pytest.mark.parametrize(
    "role",
    [
        UserRole.CUSTOMER,
        UserRole.SELLER,
        UserRole.SUPPORT,
        UserRole.OPERATIONS,
        UserRole.ADMIN,
    ],
)
def test_all_five_roles_can_create_principal(role):
    user = UserPrincipal(
        user_id=f"TEST-{role.value.upper()}",
        role=role,
        is_active=True,
    )

    assert user.role == role
    assert user.is_active is True
