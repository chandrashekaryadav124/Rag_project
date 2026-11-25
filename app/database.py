
# ==========================================
# CUSTOMER RAG - LIVE MYSQL CONNECTION
# ==========================================

import os
from functools import lru_cache

import mysql.connector
from mysql.connector import pooling

from dotenv import load_dotenv


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(BASE_DIR, ".env"))


# ==========================================
# DATABASE CONNECTION
# ==========================================

def _build_mysql_config() -> dict:

    config = {
        "host": os.getenv("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "connection_timeout": 10,
        "autocommit": False
    }

    ssl_ca = os.getenv("MYSQL_SSL_CA")

    if ssl_ca:
        if not os.path.isabs(ssl_ca):
            ssl_ca = os.path.join(BASE_DIR, ssl_ca)

        if not os.path.isfile(ssl_ca):
            raise FileNotFoundError(
                f"MySQL CA certificate not found: {ssl_ca}"
            )

        config["ssl_ca"] = ssl_ca
        config["ssl_verify_cert"] = True
        config["ssl_verify_identity"] = True

    required = [
        "host",
        "user",
        "password",
        "database"
    ]

    missing = [
        key for key in required
        if not config.get(key)
    ]

    if missing:
        raise ValueError(
            "Missing MySQL configuration: "
            + ", ".join(missing)
        )

    return config


@lru_cache(maxsize=1)
def _get_mysql_pool() -> pooling.MySQLConnectionPool:
    config = _build_mysql_config()
    pool_size = max(1, int(os.getenv("MYSQL_POOL_SIZE", "5")))
    return pooling.MySQLConnectionPool(
        pool_name="rag_mysql_pool",
        pool_size=pool_size,
        pool_reset_session=True,
        **config,
    )


def get_db_connection():
    return _get_mysql_pool().get_connection()


# ==========================================
# DATABASE CONNECTION TEST
# ==========================================

def test_db_connection():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                DATABASE() AS database_name,
                CURRENT_USER() AS connected_user,
                VERSION() AS mysql_version
        """)

        result = cursor.fetchone()

        print("\nMYSQL CONNECTION SUCCESSFUL")
        print("Database:", result["database_name"])
        print("Connected user:", result["connected_user"])
        print("MySQL version:", result["mysql_version"])

        return result

    except Exception as error:

        print("\nMYSQL CONNECTION ERROR:", str(error))

        raise

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


def check_mysql_health() -> dict:
    """Run a lightweight MySQL connectivity check for /health."""

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return {"status": "connected"}
    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    test_db_connection()
