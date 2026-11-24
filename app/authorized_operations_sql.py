
from app.database import get_db_connection
from app.core.security import UserPrincipal, UserRole


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
