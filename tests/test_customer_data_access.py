
from app.authorized_customer_sql import get_own_customer_orders
from app.core.security import UserPrincipal, UserRole


class FakeCursor:

    def __init__(self, results=None):
        self.results = results or []
        self.executed_query = None
        self.executed_params = None

    def execute(self, query, params):
        self.executed_query = query
        self.executed_params = params

    def fetchall(self):
        return self.results

    def close(self):
        pass


class FakeConnection:

    def __init__(self, results=None):
        self.fake_cursor = FakeCursor(results)
        self.closed = False

    def cursor(self, dictionary=True):
        return self.fake_cursor

    def close(self):
        self.closed = True


def make_customer(user_id="CUSTOMER-TEST-01"):
    return UserPrincipal(
        user_id=user_id,
        role=UserRole.CUSTOMER,
        is_active=True
    )


def test_customer_can_access_own_orders(monkeypatch):

    expected_orders = [
        {"order_id": "ORD-001", "order_status": "delivered"}
    ]

    connection = FakeConnection(expected_orders)

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: connection
    )

    customer = make_customer()

    result = get_own_customer_orders(customer)

    assert result == expected_orders

    assert connection.fake_cursor.executed_params == (
        "CUSTOMER-TEST-01",
        "CUSTOMER-TEST-01",
        10
    )


def test_customer_query_filters_by_authenticated_user(monkeypatch):

    connection = FakeConnection()

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: connection
    )

    customer = make_customer("CUSTOMER-A")

    get_own_customer_orders(customer)

    query = connection.fake_cursor.executed_query

    assert "o.customer_id = %s" in query
    assert "c.app_user_id = %s" in query

    assert connection.fake_cursor.executed_params == (
        "CUSTOMER-A",
        "CUSTOMER-A",
        10
    )


def test_customer_with_no_orders_gets_empty_list(monkeypatch):

    connection = FakeConnection([])

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: connection
    )

    result = get_own_customer_orders(make_customer())

    assert result == []


def test_customer_order_limit_is_bounded(monkeypatch):

    connection = FakeConnection()

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: connection
    )

    get_own_customer_orders(make_customer(), limit=100)

    assert connection.fake_cursor.executed_params == (
        "CUSTOMER-TEST-01",
        "CUSTOMER-TEST-01",
        20
    )


def test_seller_cannot_access_customer_orders(monkeypatch):

    seller = UserPrincipal(
        user_id="SELLER-01",
        role=UserRole.SELLER,
        is_active=True
    )

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: None
    )

    try:
        get_own_customer_orders(seller)
        assert False, "Expected PermissionError"
    except PermissionError:
        pass


def test_inactive_customer_cannot_access_orders(monkeypatch):

    # UserPrincipal itself rejects inactive users in this project.
    # This test verifies that rejection happens before DB access.

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: None
    )

    try:
        UserPrincipal(
            user_id="CUSTOMER-01",
            role=UserRole.CUSTOMER,
            is_active=False
        )
        assert False, "Expected PermissionError"
    except PermissionError:
        pass


def test_database_connection_failure_is_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: None
    )

    try:
        get_own_customer_orders(make_customer())
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_customer_connection_is_closed(monkeypatch):

    connection = FakeConnection([])

    monkeypatch.setattr(
        "app.authorized_customer_sql.get_db_connection",
        lambda: connection
    )

    get_own_customer_orders(make_customer())

    assert connection.closed is True
