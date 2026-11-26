
from app.core.analytics_helpers import attach_yoy_deltas, to_number
from app.core.security import UserPrincipal, UserRole
from app.database import get_db_connection


def get_own_seller_products(
    principal: UserPrincipal,
    limit: int = 20
):
    """
    Retrieve products belonging only to the authenticated seller.
    """

    if principal.role != UserRole.SELLER:
        raise PermissionError(
            "Only sellers can access seller product records."
        )

    if not principal.is_active:
        raise PermissionError("Inactive users cannot access products.")

    limit = max(1, min(int(limit), 50))

    connection = get_db_connection()

    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                sp.seller_product_id,
                sp.product_id,
                p.product_name,
                sp.sku,
                sp.selling_price,
                sp.discount_price,
                sp.stock_quantity,
                sp.listing_status
            FROM seller_products AS sp
            INNER JOIN sellers AS s
                ON s.seller_id = sp.seller_id
            INNER JOIN products AS p
                ON p.product_id = sp.product_id
            WHERE s.user_id = %s
              AND s.approval_status = 'approved'
            ORDER BY sp.created_at DESC
            LIMIT %s
        """

        cursor.execute(
            query,
            (principal.user_id, limit)
        )

        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()

        connection.close()


def get_own_seller_analytics(principal: UserPrincipal) -> dict:
    """
    Retrieve aggregate store orders, distinct customer count, and multi-year
    revenue with YoY comparison for the authenticated approved seller.
    Never returns individual customer PII or other sellers' metrics.
    """
    if principal.role != UserRole.SELLER:
        raise PermissionError(
            "Only sellers can access seller store analytics."
        )

    if not principal.is_active:
        raise PermissionError("Inactive users cannot access seller analytics.")

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT so.seller_order_id) AS total_orders,
                COUNT(DISTINCT o.customer_id) AS distinct_customers,
                COALESCE(SUM(CASE WHEN LOWER(COALESCE(so.status, '')) <> 'cancelled' THEN so.subtotal ELSE 0 END), 0) AS active_revenue_inr,
                COALESCE(SUM(so.subtotal), 0) AS gross_revenue_inr,
                SUM(CASE WHEN LOWER(COALESCE(so.status, '')) = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
            FROM seller_orders AS so
            INNER JOIN sellers AS s
                ON s.seller_id = so.seller_id
            INNER JOIN orders AS o
                ON o.order_id = so.order_id
            WHERE s.user_id = %s
              AND s.approval_status = 'approved'
            """,
            (principal.user_id,),
        )
        overall_row = cursor.fetchone() or {}

        cursor.execute(
            """
            SELECT
                LEFT(o.order_date, 4) AS year,
                COUNT(DISTINCT so.seller_order_id) AS orders_count,
                COUNT(DISTINCT o.customer_id) AS distinct_customers,
                COALESCE(SUM(CASE WHEN LOWER(COALESCE(so.status, '')) <> 'cancelled' THEN so.subtotal ELSE 0 END), 0) AS revenue_inr
            FROM seller_orders AS so
            INNER JOIN sellers AS s
                ON s.seller_id = so.seller_id
            INNER JOIN orders AS o
                ON o.order_id = so.order_id
            WHERE s.user_id = %s
              AND s.approval_status = 'approved'
              AND o.order_date IS NOT NULL
              AND LENGTH(o.order_date) >= 4
            GROUP BY LEFT(o.order_date, 4)
            ORDER BY year DESC
            LIMIT 5
            """,
            (principal.user_id,),
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

        cursor.execute(
            """
            SELECT
                so.status,
                COUNT(DISTINCT so.seller_order_id) AS orders_count,
                COALESCE(SUM(so.subtotal), 0) AS subtotal_inr
            FROM seller_orders AS so
            INNER JOIN sellers AS s
                ON s.seller_id = so.seller_id
            WHERE s.user_id = %s
              AND s.approval_status = 'approved'
            GROUP BY so.status
            ORDER BY orders_count DESC
            """,
            (principal.user_id,),
        )
        status_rows = [
            {
                "status": row.get("status"),
                "orders_count": to_number(row.get("orders_count"), as_int=True),
                "subtotal_inr": to_number(row.get("subtotal_inr")),
            }
            for row in (cursor.fetchall() or [])
        ]

        return {
            "store_summary": {
                "total_orders": to_number(overall_row.get("total_orders"), as_int=True),
                "distinct_customers": to_number(overall_row.get("distinct_customers"), as_int=True),
                "active_revenue_inr": to_number(overall_row.get("active_revenue_inr")),
                "gross_revenue_inr": to_number(overall_row.get("gross_revenue_inr")),
                "cancelled_orders": to_number(overall_row.get("cancelled_orders"), as_int=True),
            },
            "yearly_performance": attach_yoy_deltas(
                yearly_rows,
                metric_keys=("revenue_inr", "orders_count", "distinct_customers"),
            ),
            "order_status_breakdown": status_rows,
        }
    finally:
        if cursor:
            cursor.close()
        connection.close()
