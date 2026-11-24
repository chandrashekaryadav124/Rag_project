
from app.core.security import UserRole

# Initial RAG access policy.
# These permissions are a development baseline and must be
# checked against the Phase I business rules before production.

ROLE_PERMISSIONS = {
    UserRole.CUSTOMER: {
        "product_search",
        "product_knowledge",
        "own_orders",
        "own_returns",
        "own_refunds",
        "own_support_tickets",
    },

    UserRole.SELLER: {
        "own_products",
        "own_inventory",
        "own_seller_orders",
        "product_knowledge",
    },

    UserRole.SUPPORT: {
        "product_search",
        "product_knowledge",
        "customer_support",
        "order_lookup",
        "return_lookup",
        "refund_lookup",
    },

    UserRole.OPERATIONS: {
        "product_search",
        "product_knowledge",
        "inventory_overview",
        "shipment_lookup",
        "delivery_analytics",
    },

    UserRole.ADMIN: {
        "product_search",
        "product_knowledge",
        "own_orders",
        "own_returns",
        "own_refunds",
        "own_support_tickets",
        "own_products",
        "own_inventory",
        "own_seller_orders",
        "customer_support",
        "order_lookup",
        "return_lookup",
        "refund_lookup",
        "inventory_overview",
        "shipment_lookup",
        "delivery_analytics",
        "system_analytics",
        "system_management",
    },
}


def has_permission(role: UserRole, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())


def get_role_permissions(role: UserRole) -> list:
    return sorted(ROLE_PERMISSIONS.get(role, set()))
