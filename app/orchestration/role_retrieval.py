
from app.core.security import UserPrincipal, UserRole

from app.authorized_customer_sql import get_own_customer_orders
from app.authorized_seller_sql import get_own_seller_products

from app.authorized_support_sql import (
    get_support_order_summary,
    get_support_tickets,
    get_support_tickets_by_order,
)

from app.authorized_operations_sql import (
    get_shipment_summary,
    get_inventory_snapshot_summary,
)

from app.authorized_admin_sql import get_system_analytics


def retrieve_role_data(
    principal: UserPrincipal,
    task: str,
    order_id: str = None,
    limit: int = 10,
):
    """
    Central dispatcher for predefined, role-authorized SQL tools.

    Only predefined tasks are supported.
    No user-generated SQL is executed.
    """

    if not principal.is_active:
        raise PermissionError("Inactive users cannot retrieve data.")

    if not isinstance(task, str):
        raise ValueError("Invalid retrieval task.")

    task = task.strip().lower()

    # CUSTOMER
    if principal.role == UserRole.CUSTOMER:
        if task == "own_orders":
            return get_own_customer_orders(principal, limit)

    # SELLER
    elif principal.role == UserRole.SELLER:
        if task == "own_products":
            return get_own_seller_products(principal, limit)

    # SUPPORT
    elif principal.role == UserRole.SUPPORT:
        if task == "order_summary":
            return get_support_order_summary(principal, order_id)

        if task == "support_tickets":
            return get_support_tickets(principal, limit)

        if task == "tickets_by_order":
            return get_support_tickets_by_order(principal, order_id)

    # OPERATIONS
    elif principal.role == UserRole.OPERATIONS:
        if task == "shipment_summary":
            return get_shipment_summary(principal, order_id)

        if task == "inventory_overview":
            return get_inventory_snapshot_summary(principal)

    # ADMIN
    elif principal.role == UserRole.ADMIN:
        if task == "system_analytics":
            return get_system_analytics(principal)

        if task == "support_tickets":
            return get_support_tickets(principal, limit)

        if task == "order_summary":
            return get_support_order_summary(principal, order_id)

        if task == "tickets_by_order":
            return get_support_tickets_by_order(principal, order_id)

        if task == "shipment_summary":
            return get_shipment_summary(principal, order_id)

        if task == "inventory_overview":
            return get_inventory_snapshot_summary(principal)

    raise PermissionError(
        f"Task '{task}' is not permitted for role '{principal.role.value}'."
    )
