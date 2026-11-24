
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_principal
from app.core.permissions import require_roles
from app.core.security import UserPrincipal, UserRole

from app.authorized_customer_sql import get_own_customer_orders
from app.authorized_seller_sql import get_own_seller_products

from app.authorized_support_sql import (
    get_support_order_summary,
    get_support_tickets,
)

from app.authorized_operations_sql import (
    get_shipment_summary,
    get_inventory_snapshot_summary,
)

from app.authorized_admin_sql import get_system_analytics


router = APIRouter(
    prefix="/api/v1/role-data",
    tags=["Role-Based Data Access"],
)


@router.get("/customer/orders")
def customer_orders(
    user: UserPrincipal = Depends(
        require_roles(UserRole.CUSTOMER)
    ),
):
    return get_own_customer_orders(user)


@router.get("/seller/products")
def seller_products(
    user: UserPrincipal = Depends(
        require_roles(UserRole.SELLER)
    ),
):
    return get_own_seller_products(user)


@router.get("/support/orders/{order_id}")
def support_order(
    order_id: str,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SUPPORT, UserRole.ADMIN)
    ),
):
    return get_support_order_summary(user, order_id)


@router.get("/support/tickets")
def support_tickets(
    limit: int = 20,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SUPPORT, UserRole.ADMIN)
    ),
):
    return get_support_tickets(user, limit)


@router.get("/operations/shipments/{order_id}")
def operations_shipments(
    order_id: str,
    user: UserPrincipal = Depends(
        require_roles(UserRole.OPERATIONS, UserRole.ADMIN)
    ),
):
    return get_shipment_summary(user, order_id)


@router.get("/operations/inventory")
def operations_inventory(
    user: UserPrincipal = Depends(
        require_roles(UserRole.OPERATIONS, UserRole.ADMIN)
    ),
):
    return get_inventory_snapshot_summary(user)


@router.get("/admin/analytics")
def admin_analytics(
    user: UserPrincipal = Depends(
        require_roles(UserRole.ADMIN)
    ),
):
    return get_system_analytics(user)
