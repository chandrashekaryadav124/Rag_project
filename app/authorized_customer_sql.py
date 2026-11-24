
from app.database import get_db_connection
from app.core.security import UserPrincipal, UserRole


def get_own_customer_orders(
    principal: UserPrincipal,
    limit: int = 10
):
    """
    Retrieve orders belonging only to the authenticated customer.

    Supports two existing order relationships:
    1. Direct mapping: orders.customer_id = app_users.user_id
    2. Profile mapping: orders.customer_id = customers.customer_id
       and customers.app_user_id = app_users.user_id

    The user ID comes exclusively from the verified authentication principal.
    """

    if principal.role != UserRole.CUSTOMER:
        raise PermissionError(
            "Only customers can access their own customer orders."
        )

    if not principal.is_active:
        raise PermissionError(
            "Inactive users cannot access orders."
        )

    limit = max(1, min(int(limit), 20))

    connection = get_db_connection()

    if connection is None:
        raise RuntimeError(
            "Database connection unavailable."
        )

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        # Retrieve orders through either authorized relationship.
        query = """
            SELECT
                o.order_id,
                o.order_date,
                o.order_status,
                o.shipping_method,
                o.final_amount
            FROM orders AS o
            WHERE
                o.customer_id = %s
                OR EXISTS (
                    SELECT 1
                    FROM customers AS c
                    WHERE c.customer_id = o.customer_id
                      AND c.app_user_id = %s
                )
            ORDER BY o.order_date DESC
            LIMIT %s
        """

        cursor.execute(
            query,
            (
                principal.user_id,
                principal.user_id,
                limit
            )
        )

        orders = cursor.fetchall()

        print(
            f"[ORDER RETRIEVAL DEBUG] "
            f"role={principal.role.value}, "
            f"orders_retrieved={len(orders)}"
        )

        return orders

    finally:
        if cursor:
            cursor.close()

        connection.close()
