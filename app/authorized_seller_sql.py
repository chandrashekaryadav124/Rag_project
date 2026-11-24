
from app.database import get_db_connection
from app.core.security import UserPrincipal, UserRole


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
