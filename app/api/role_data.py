
from fastapi import APIRouter, Depends, HTTPException

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


def _execute_role_query(query_fn, *args, **kwargs):
    """Execute a role-authorized SQL query with consistent HTTP error mapping."""
    try:
        return query_fn(*args, **kwargs)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        print("Role data access error:", str(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve authorized role data.",
        ) from error


@router.get("/customer/orders")
def customer_orders(
    user: UserPrincipal = Depends(
        require_roles(UserRole.CUSTOMER)
    ),
):
    return _execute_role_query(get_own_customer_orders, user)


@router.get("/seller/products")
def seller_products(
    user: UserPrincipal = Depends(
        require_roles(UserRole.SELLER)
    ),
):
    return _execute_role_query(get_own_seller_products, user)


@router.get("/support/orders/{order_id}")
def support_order(
    order_id: str,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SUPPORT, UserRole.ADMIN)
    ),
):
    return _execute_role_query(get_support_order_summary, user, order_id)


@router.get("/support/tickets")
def support_tickets(
    limit: int = 20,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SUPPORT, UserRole.ADMIN)
    ),
):
    return _execute_role_query(get_support_tickets, user, limit)


@router.get("/operations/shipments/{order_id}")
def operations_shipments(
    order_id: str,
    user: UserPrincipal = Depends(
        require_roles(UserRole.OPERATIONS, UserRole.ADMIN)
    ),
):
    return _execute_role_query(get_shipment_summary, user, order_id)


@router.get("/operations/inventory")
def operations_inventory(
    user: UserPrincipal = Depends(
        require_roles(UserRole.OPERATIONS, UserRole.ADMIN)
    ),
):
    return _execute_role_query(get_inventory_snapshot_summary, user)


@router.get("/admin/analytics")
def admin_analytics(
    user: UserPrincipal = Depends(
        require_roles(UserRole.ADMIN)
    ),
):
    return _execute_role_query(get_system_analytics, user)
