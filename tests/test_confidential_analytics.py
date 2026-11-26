import pytest

from app.authorized_admin_sql import get_admin_yearly_analytics
from app.authorized_customer_sql import get_own_customer_analytics
from app.authorized_operations_sql import get_operations_delivery_analytics
from app.authorized_seller_sql import get_own_seller_analytics
from app.authorized_support_sql import get_support_analytics
from app.core.analytics_helpers import attach_yoy_deltas
from app.core.security import UserPrincipal, UserRole
from app.orchestration.role_retrieval import retrieve_role_data


def _user(role: UserRole, user_id: str = "USR-TEST") -> UserPrincipal:
    return UserPrincipal(user_id=user_id, role=role, is_active=True)


def test_attach_yoy_deltas_computes_previous_year_differences():
    rows = [
        {"year": "2024", "revenue_inr": 1000.0, "orders_count": 10},
        {"year": "2025", "revenue_inr": 1500.0, "orders_count": 15},
        {"year": "2023", "revenue_inr": 800.0, "orders_count": 8},
    ]

    result = attach_yoy_deltas(rows, metric_keys=("revenue_inr", "orders_count"))

    assert [r["year"] for r in result] == ["2025", "2024", "2023"]
    assert result[0]["previous_year"] == "2024"
    assert result[0]["revenue_inr_vs_prev_year"] == 500.0
    assert result[0]["revenue_inr_yoy_pct"] == 50.0
    assert result[1]["previous_year"] == "2023"
    assert result[1]["orders_count_vs_prev_year"] == 2
    assert result[2]["previous_year"] is None
    assert result[2]["revenue_inr_vs_prev_year"] is None


@pytest.mark.parametrize(
    "caller_role, fn",
    [
        (UserRole.SELLER, get_own_customer_analytics),
        (UserRole.SUPPORT, get_own_customer_analytics),
        (UserRole.CUSTOMER, get_own_seller_analytics),
        (UserRole.SUPPORT, get_own_seller_analytics),
        (UserRole.CUSTOMER, get_support_analytics),
        (UserRole.SELLER, get_support_analytics),
        (UserRole.OPERATIONS, get_support_analytics),
        (UserRole.CUSTOMER, get_operations_delivery_analytics),
        (UserRole.SELLER, get_operations_delivery_analytics),
        (UserRole.SUPPORT, get_operations_delivery_analytics),
        (UserRole.CUSTOMER, get_admin_yearly_analytics),
        (UserRole.SELLER, get_admin_yearly_analytics),
        (UserRole.SUPPORT, get_admin_yearly_analytics),
        (UserRole.OPERATIONS, get_admin_yearly_analytics),
    ],
)
def test_confidential_analytics_functions_enforce_strict_rbac(caller_role, fn):
    with pytest.raises(PermissionError):
        fn(_user(caller_role))


def test_seller_analytics_queries_only_approved_own_seller_without_pii(monkeypatch):
    captured_queries = []

    class FakeCursor:
        def __init__(self):
            self._call = 0

        def execute(self, query, params=None):
            captured_queries.append((query, params))
            self._call += 1

        def fetchone(self):
            return {
                "total_orders": 20,
                "distinct_customers": 15,
                "active_revenue_inr": 50000.0,
                "gross_revenue_inr": 52000.0,
                "cancelled_orders": 1,
            }

        def fetchall(self):
            if self._call == 2:
                return [
                    {"year": "2025", "orders_count": 12, "distinct_customers": 10, "revenue_inr": 30000.0},
                    {"year": "2024", "orders_count": 8, "distinct_customers": 7, "revenue_inr": 20000.0},
                ]
            return [{"status": "delivered", "orders_count": 19, "subtotal_inr": 50000.0}]

        def close(self):
            pass

    class FakeConn:
        def cursor(self, dictionary=True):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr("app.authorized_seller_sql.get_db_connection", lambda: FakeConn())

    result = get_own_seller_analytics(_user(UserRole.SELLER, user_id="USR-SELLER-001"))

    assert result["store_summary"]["distinct_customers"] == 15
    assert result["yearly_performance"][0]["revenue_inr_vs_prev_year"] == 10000.0
    assert result["yearly_performance"][0]["revenue_inr_yoy_pct"] == 50.0
    assert all(params == ("USR-SELLER-001",) for _, params in captured_queries)
    assert all("approval_status = 'approved'" in q for q, _ in captured_queries)
    assert "customer_name" not in str(result)
    assert "recipient_phone" not in str(result)


@pytest.mark.parametrize(
    "role, disallowed_task",
    [
        (UserRole.CUSTOMER, "seller_hybrid_overview"),
        (UserRole.SELLER, "customer_hybrid_overview"),
        (UserRole.SELLER, "admin_hybrid_overview"),
        (UserRole.SUPPORT, "operations_hybrid_overview"),
        (UserRole.OPERATIONS, "support_hybrid_overview"),
    ],
)
def test_composite_hybrid_tasks_reject_cross_role_access(role, disallowed_task):
    with pytest.raises(PermissionError):
        retrieve_role_data(_user(role), disallowed_task)
