
from app.core.analytics_helpers import attach_yoy_deltas, to_number
from app.core.security import UserPrincipal, UserRole
from app.database import get_db_connection


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


def get_own_customer_analytics(principal: UserPrincipal) -> dict:
    """
    Retrieve authenticated customer's aggregate spending, yearly YoY trend,
    and own return/refund records. Strictly isolated to principal.user_id.
    """
    if principal.role != UserRole.CUSTOMER:
        raise PermissionError(
            "Only customers can access their own customer analytics."
        )

    if not principal.is_active:
        raise PermissionError(
            "Inactive users cannot access customer analytics."
        )

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        ownership_clause = """
            (
                o.customer_id = %s
                OR EXISTS (
                    SELECT 1
                    FROM customers AS c
                    WHERE c.customer_id = o.customer_id
                      AND c.app_user_id = %s
                )
            )
        """

        cursor.execute(
            f"""
            SELECT
                COUNT(DISTINCT o.order_id) AS total_orders,
                COALESCE(SUM(CASE WHEN LOWER(COALESCE(o.order_status, '')) <> 'cancelled' THEN o.final_amount ELSE 0 END), 0) AS total_spent_inr,
                SUM(CASE WHEN LOWER(COALESCE(o.order_status, '')) = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
            FROM orders AS o
            WHERE {ownership_clause}
            """,
            (principal.user_id, principal.user_id),
        )
        summary_row = cursor.fetchone() or {}

        cursor.execute(
            f"""
            SELECT
                LEFT(o.order_date, 4) AS year,
                COUNT(DISTINCT o.order_id) AS orders_count,
                COALESCE(SUM(CASE WHEN LOWER(COALESCE(o.order_status, '')) <> 'cancelled' THEN o.final_amount ELSE 0 END), 0) AS spent_inr
            FROM orders AS o
            WHERE {ownership_clause}
              AND o.order_date IS NOT NULL
              AND LENGTH(o.order_date) >= 4
            GROUP BY LEFT(o.order_date, 4)
            ORDER BY year DESC
            LIMIT 5
            """,
            (principal.user_id, principal.user_id),
        )
        yearly_rows = [
            {
                "year": row["year"],
                "orders_count": to_number(row.get("orders_count"), as_int=True),
                "spent_inr": to_number(row.get("spent_inr")),
            }
            for row in (cursor.fetchall() or [])
        ]

        cursor.execute(
            """
            SELECT
                r.return_id,
                r.order_id,
                r.return_date,
                r.return_reason,
                r.refund_amount,
                r.return_status
            FROM returns AS r
            INNER JOIN orders AS o
                ON o.order_id = r.order_id
            WHERE
                o.customer_id = %s
                OR EXISTS (
                    SELECT 1
                    FROM customers AS c
                    WHERE c.customer_id = o.customer_id
                      AND c.app_user_id = %s
                )
            ORDER BY r.return_date DESC
            LIMIT 10
            """,
            (principal.user_id, principal.user_id),
        )
        own_returns = cursor.fetchall() or []

        return {
            "summary": {
                "total_orders": to_number(summary_row.get("total_orders"), as_int=True),
                "total_spent_inr": to_number(summary_row.get("total_spent_inr")),
                "cancelled_orders": to_number(summary_row.get("cancelled_orders"), as_int=True),
            },
            "yearly_spending": attach_yoy_deltas(
                yearly_rows,
                metric_keys=("spent_inr", "orders_count"),
            ),
            "own_returns": own_returns,
        }
    finally:
        if cursor:
            cursor.close()
        connection.close()
