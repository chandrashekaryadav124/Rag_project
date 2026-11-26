
from app.core.analytics_helpers import attach_yoy_deltas, to_number
from app.core.security import UserPrincipal, UserRole
from app.database import get_db_connection


def _check_support(principal):
    if not principal.is_active:
        raise PermissionError("Inactive users cannot access support records.")

    if principal.role not in {UserRole.SUPPORT, UserRole.ADMIN}:
        raise PermissionError("Support access is not authorized.")


def get_support_order_summary(
    principal: UserPrincipal,
    order_id: str
):
    _check_support(principal)

    if not order_id or not order_id.strip():
        raise ValueError("Order ID is required.")

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                o.order_id,
                o.customer_id,
                o.order_date,
                o.order_status,
                o.shipping_method,
                o.final_amount,
                o.delivery_city,
                o.delivery_state
            FROM orders AS o
            WHERE o.order_id = %s
            LIMIT 1
        """

        cursor.execute(query, (order_id.strip(),))
        return cursor.fetchone()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_support_tickets(
    principal: UserPrincipal,
    limit: int = 20
):
    _check_support(principal)

    limit = max(1, min(int(limit), 50))

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                ticket_id,
                customer_id,
                order_id,
                issue_type,
                customer_message,
                resolution_guidance,
                ticket_status,
                created_at,
                source_type
            FROM support_tickets
            ORDER BY created_at DESC
            LIMIT %s
        """

        cursor.execute(query, (limit,))
        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_support_tickets_by_order(
    principal: UserPrincipal,
    order_id: str
):
    _check_support(principal)

    if not order_id or not order_id.strip():
        raise ValueError("Order ID is required.")

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                ticket_id,
                customer_id,
                order_id,
                issue_type,
                customer_message,
                resolution_guidance,
                ticket_status,
                created_at,
                source_type
            FROM support_tickets
            WHERE order_id = %s
            ORDER BY created_at DESC
            LIMIT 50
        """

        cursor.execute(query, (order_id.strip(),))
        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_support_analytics(principal: UserPrincipal) -> dict:
    """
    Retrieve aggregate support ticket counts by status, issue_type, and year
    with YoY comparison. Authorized for SUPPORT and ADMIN roles only.
    """
    _check_support(principal)

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                ticket_status,
                COUNT(*) AS ticket_count
            FROM support_tickets
            GROUP BY ticket_status
            ORDER BY ticket_count DESC
            """
        )
        status_breakdown = [
            {
                "ticket_status": row.get("ticket_status"),
                "ticket_count": to_number(row.get("ticket_count"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        cursor.execute(
            """
            SELECT
                issue_type,
                COUNT(*) AS ticket_count
            FROM support_tickets
            GROUP BY issue_type
            ORDER BY ticket_count DESC
            LIMIT 15
            """
        )
        issue_breakdown = [
            {
                "issue_type": row.get("issue_type"),
                "ticket_count": to_number(row.get("ticket_count"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        cursor.execute(
            """
            SELECT
                LEFT(created_at, 4) AS year,
                COUNT(*) AS ticket_count
            FROM support_tickets
            WHERE created_at IS NOT NULL
              AND LENGTH(created_at) >= 4
            GROUP BY LEFT(created_at, 4)
            ORDER BY year DESC
            LIMIT 5
            """
        )
        yearly_rows = [
            {
                "year": row["year"],
                "ticket_count": to_number(row.get("ticket_count"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        return {
            "status_breakdown": status_breakdown,
            "issue_type_breakdown": issue_breakdown,
            "yearly_tickets": attach_yoy_deltas(
                yearly_rows,
                metric_keys=("ticket_count",),
            ),
        }
    finally:
        if cursor:
            cursor.close()
        connection.close()
