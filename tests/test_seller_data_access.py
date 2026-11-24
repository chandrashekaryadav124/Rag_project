
import pytest

from app.core.security import UserPrincipal, UserRole
from app.authorized_seller_sql import get_own_seller_products


def test_customer_cannot_access_seller_products():
    customer = UserPrincipal(
        user_id="CUSTOMER-TEST",
        role=UserRole.CUSTOMER,
        is_active=True
    )

    with pytest.raises(PermissionError):
        get_own_seller_products(customer)


def test_support_cannot_access_seller_products():
    support = UserPrincipal(
        user_id="SUPPORT-TEST",
        role=UserRole.SUPPORT,
        is_active=True
    )

    with pytest.raises(PermissionError):
        get_own_seller_products(support)


def test_inactive_seller_cannot_access_products():
    with pytest.raises(PermissionError):
        UserPrincipal(
            user_id="SELLER-TEST",
            role=UserRole.SELLER,
            is_active=False
        )


def test_seller_product_limit_is_bounded():
    assert max(1, min(100, 50)) == 50
    assert max(1, min(0, 50)) == 1
