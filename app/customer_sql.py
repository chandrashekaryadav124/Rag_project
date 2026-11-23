
# ==========================================
# CUSTOMER RAG - LIVE SQL RETRIEVAL
# ==========================================

from app.database import get_db_connection


# ==========================================
# 1. SEARCH PRODUCTS
# ==========================================

def search_products(search_term, limit=5):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                product_id,
                product_name,
                description,
                category,
                subcategory,
                brand,
                price,
                discount_range,
                rating_average,
                rating_count,
                stock_quantity
            FROM products
            WHERE product_name LIKE %s
               OR category LIKE %s
               OR brand LIKE %s
            LIMIT %s
        """

        search_value = f"%{search_term}%"

        cursor.execute(
            query,
            (
                search_value,
                search_value,
                search_value,
                limit
            )
        )

        return cursor.fetchall()

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 2. CHECK PRODUCT AVAILABILITY
# ==========================================

def check_product_availability(product_name):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                product_id,
                product_name,
                stock_quantity,
                price
            FROM products
            WHERE product_name LIKE %s
            LIMIT 5
        """

        cursor.execute(
            query,
            (f"%{product_name}%",)
        )

        return cursor.fetchall()

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 3. GET LOGGED-IN CUSTOMER ORDERS
# ==========================================

def get_customer_orders(user_id, limit=10):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                o.order_id,
                o.order_date,
                o.order_status,
                o.shipping_method,
                o.final_amount
            FROM orders o

            INNER JOIN customers c
                ON c.customer_id = o.customer_id

            INNER JOIN app_users u
                ON u.user_id = c.app_user_id

            WHERE u.user_id = %s

            ORDER BY o.order_date DESC

            LIMIT %s
        """

        cursor.execute(
            query,
            (user_id, limit)
        )

        return cursor.fetchall()

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 4. TEST CUSTOMER SQL RETRIEVAL
# ==========================================

if __name__ == "__main__":

    print("\nCUSTOMER SQL RETRIEVAL TEST")

    products = search_products("phone")

    print("\nPRODUCT RESULTS:")

    for product in products:
        print(product)

    print("\nSQL RETRIEVAL TEST COMPLETED")
