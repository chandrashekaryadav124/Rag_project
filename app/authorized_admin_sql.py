
from app.database import get_db_connection
from app.core.security import UserPrincipal, UserRole


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
