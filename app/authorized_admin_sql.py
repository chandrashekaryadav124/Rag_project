
from app.core.analytics_helpers import attach_yoy_deltas, to_number
from app.core.security import UserPrincipal, UserRole
from app.database import get_db_connection


def get_system_analytics(principal: UserPrincipal):
    if not principal.is_active:
        raise PermissionError("Inactive users cannot access analytics.")

    if principal.role != UserRole.ADMIN:
        raise PermissionError("Admin access is required.")

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                (SELECT COUNT(*) FROM customers) AS total_customers,
                (SELECT COUNT(*) FROM orders) AS total_orders,
                (SELECT COUNT(*) FROM products) AS total_products,
                (SELECT COUNT(*) FROM returns) AS total_returns,
                (SELECT COUNT(*) FROM support_tickets) AS total_support_tickets,
                (SELECT COUNT(*) FROM sellers) AS total_sellers
        """

        cursor.execute(query)
        return cursor.fetchone()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_admin_yearly_analytics(principal: UserPrincipal) -> dict:
    """
    Retrieve platform-wide multi-year orders, distinct customers, revenue (INR),
    and returns with YoY comparisons. Strictly restricted to ADMIN.
    """
    if not principal.is_active:
        raise PermissionError("Inactive users cannot access analytics.")

    if principal.role != UserRole.ADMIN:
        raise PermissionError("Admin access is required.")

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM customers) AS total_customers,
                (SELECT COUNT(*) FROM orders) AS total_orders,
                (SELECT COUNT(*) FROM products) AS total_products,
                (SELECT COUNT(*) FROM returns) AS total_returns,
                (SELECT COUNT(*) FROM support_tickets) AS total_support_tickets,
                (SELECT COUNT(*) FROM sellers) AS total_sellers,
                (SELECT COALESCE(SUM(CASE WHEN LOWER(COALESCE(order_status, '')) <> 'cancelled' THEN final_amount ELSE 0 END), 0) FROM orders) AS total_revenue_inr
            """
        )
        totals_row = cursor.fetchone() or {}

        cursor.execute(
            """
            SELECT
                LEFT(order_date, 4) AS year,
                COUNT(DISTINCT order_id) AS orders_count,
                COUNT(DISTINCT customer_id) AS distinct_customers,
                COALESCE(SUM(CASE WHEN LOWER(COALESCE(order_status, '')) <> 'cancelled' THEN final_amount ELSE 0 END), 0) AS revenue_inr
            FROM orders
            WHERE order_date IS NOT NULL
              AND LENGTH(order_date) >= 4
            GROUP BY LEFT(order_date, 4)
            ORDER BY year DESC
            LIMIT 5
            """
        )
        yearly_rows = [
            {
                "year": row["year"],
                "orders_count": to_number(row.get("orders_count"), as_int=True),
                "distinct_customers": to_number(row.get("distinct_customers"), as_int=True),
                "revenue_inr": to_number(row.get("revenue_inr")),
            }
            for row in (cursor.fetchall() or [])
        ]

        return {
            "platform_totals": {
                "total_customers": to_number(totals_row.get("total_customers"), as_int=True),
                "total_orders": to_number(totals_row.get("total_orders"), as_int=True),
                "total_products": to_number(totals_row.get("total_products"), as_int=True),
                "total_returns": to_number(totals_row.get("total_returns"), as_int=True),
                "total_support_tickets": to_number(totals_row.get("total_support_tickets"), as_int=True),
                "total_sellers": to_number(totals_row.get("total_sellers"), as_int=True),
                "total_revenue_inr": to_number(totals_row.get("total_revenue_inr")),
            },
            "yearly_platform_performance": attach_yoy_deltas(
                yearly_rows,
                metric_keys=("revenue_inr", "orders_count", "distinct_customers"),
            ),
        }
    finally:
        if cursor:
            cursor.close()
        connection.close()
