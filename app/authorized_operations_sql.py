
from app.core.analytics_helpers import attach_yoy_deltas, to_number
from app.core.security import UserPrincipal, UserRole
from app.database import get_db_connection


def _check_operations(principal):
    if not principal.is_active:
        raise PermissionError("Inactive users cannot access operations data.")

    if principal.role not in {UserRole.OPERATIONS, UserRole.ADMIN}:
        raise PermissionError("Operations access is not authorized.")


def get_shipment_summary(
    principal: UserPrincipal,
    order_id: str
):
    _check_operations(principal)

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
                shipment_id,
                order_id,
                warehouse_city,
                delivery_city,
                shipping_method,
                dispatch_date,
                expected_delivery_date,
                actual_delivery_date,
                delivery_status,
                delayed_flag,
                courier_name
            FROM shipments
            WHERE order_id = %s
            ORDER BY created_at DESC
            LIMIT 10
        """

        cursor.execute(query, (order_id.strip(),))
        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_inventory_snapshot_summary(
    principal: UserPrincipal
):
    _check_operations(principal)

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                snapshot_date,
                inventory_status,
                COUNT(*) AS inventory_records,
                SUM(available_quantity) AS total_available_quantity,
                SUM(reserved_quantity) AS total_reserved_quantity
            FROM inventory_history
            WHERE snapshot_date = (
                SELECT MAX(snapshot_date)
                FROM inventory_history
            )
            GROUP BY snapshot_date, inventory_status
            ORDER BY inventory_status
        """

        cursor.execute(query)
        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()
        connection.close()


def get_operations_delivery_analytics(principal: UserPrincipal) -> dict:
    """
    Retrieve aggregate shipment performance, delay counts, courier breakdown,
    and yearly delivery metrics with YoY comparison for OPERATIONS and ADMIN.
    """
    _check_operations(principal)

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError("Database connection unavailable.")

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_shipments,
                SUM(CASE WHEN delayed_flag = 1 THEN 1 ELSE 0 END) AS delayed_shipments
            FROM shipments
            """
        )
        overall_row = cursor.fetchone() or {}
        total_shipments = to_number(overall_row.get("total_shipments"), as_int=True)
        delayed_shipments = to_number(overall_row.get("delayed_shipments"), as_int=True)
        on_time_rate = (
            round(((total_shipments - delayed_shipments) / total_shipments) * 100, 2)
            if total_shipments > 0
            else 0.0
        )

        cursor.execute(
            """
            SELECT
                courier_name,
                COUNT(*) AS total_shipments,
                SUM(CASE WHEN delayed_flag = 1 THEN 1 ELSE 0 END) AS delayed_shipments
            FROM shipments
            GROUP BY courier_name
            ORDER BY total_shipments DESC
            LIMIT 15
            """
        )
        courier_breakdown = [
            {
                "courier_name": row.get("courier_name"),
                "total_shipments": to_number(row.get("total_shipments"), as_int=True),
                "delayed_shipments": to_number(row.get("delayed_shipments"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        cursor.execute(
            """
            SELECT
                delivery_status,
                COUNT(*) AS shipment_count
            FROM shipments
            GROUP BY delivery_status
            ORDER BY shipment_count DESC
            """
        )
        status_breakdown = [
            {
                "delivery_status": row.get("delivery_status"),
                "shipment_count": to_number(row.get("shipment_count"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        cursor.execute(
            """
            SELECT
                LEFT(o.order_date, 4) AS year,
                COUNT(*) AS total_shipments,
                SUM(CASE WHEN sh.delayed_flag = 1 THEN 1 ELSE 0 END) AS delayed_shipments
            FROM shipments AS sh
            INNER JOIN orders AS o
                ON o.order_id = sh.order_id
            WHERE o.order_date IS NOT NULL
              AND LENGTH(o.order_date) >= 4
            GROUP BY LEFT(o.order_date, 4)
            ORDER BY year DESC
            LIMIT 5
            """
        )
        yearly_rows = [
            {
                "year": row["year"],
                "total_shipments": to_number(row.get("total_shipments"), as_int=True),
                "delayed_shipments": to_number(row.get("delayed_shipments"), as_int=True),
            }
            for row in (cursor.fetchall() or [])
        ]

        return {
            "overall_shipments": {
                "total_shipments": total_shipments,
                "delayed_shipments": delayed_shipments,
                "on_time_rate_pct": on_time_rate,
            },
            "courier_breakdown": courier_breakdown,
            "status_breakdown": status_breakdown,
            "yearly_shipments": attach_yoy_deltas(
                yearly_rows,
                metric_keys=("total_shipments", "delayed_shipments"),
            ),
        }
    finally:
        if cursor:
            cursor.close()
        connection.close()
