from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _dev_headers(user_id: str, role: str) -> dict[str, str]:
    return {
        "X-Dev-User-Id": user_id,
        "X-Dev-Role": role,
    }


def test_health_returns_200_when_supabase_and_mysql_healthy(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_PROVIDER", "development")
    monkeypatch.setattr(
        "main.check_supabase_health",
        lambda: {"status": "connected", "vector_store": "supabase"},
    )
    monkeypatch.setattr(
        "main.check_mysql_health",
        lambda: {"status": "connected"},
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["vector_store"] == "supabase"
    assert body["supabase"]["status"] == "connected"
    assert body["mysql"]["status"] == "connected"


def test_health_returns_503_when_dependency_fails(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(
        "main.check_supabase_health",
        lambda: (_ for _ in ()).throw(RuntimeError("Supabase down")),
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert "Supabase down" in response.json()["detail"]


def test_health_sanitizes_503_detail_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(
        "main.check_supabase_health",
        lambda: (_ for _ in ()).throw(RuntimeError("sensitive db host error")),
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service unhealthy."


def test_seller_hybrid_chat_maps_permission_error_to_403(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_PROVIDER", "development")
    monkeypatch.setattr(
        "main.generate_hybrid_answer",
        lambda **kwargs: (_ for _ in ()).throw(
            PermissionError("Conversation session not found or not owned by this user.")
        ),
    )

    response = client.post(
        "/api/v1/seller/hybrid-chat",
        headers=_dev_headers("SELLER-01", "seller"),
        json={"question": "Show my products", "session_id": "other-session"},
    )

    assert response.status_code == 403


def test_support_hybrid_chat_maps_runtime_error_to_503(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_PROVIDER", "development")
    monkeypatch.setattr(
        "main.generate_hybrid_answer",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Database unavailable")),
    )

    response = client.post(
        "/api/v1/support/hybrid-chat",
        headers=_dev_headers("SUPPORT-01", "support"),
        json={"question": "Check order tickets", "order_id": "ORD-1"},
    )

    assert response.status_code == 503


def test_role_data_support_order_maps_blank_order_id_to_400(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_PROVIDER", "development")

    response = client.get(
        "/api/v1/role-data/support/orders/%20",
        headers=_dev_headers("SUPPORT-01", "support"),
    )

    assert response.status_code == 400


def test_legacy_chat_uses_generate_answer(monkeypatch):
    with patch(
        "main.generate_answer",
        return_value={
            "answer": "Returns are supported.",
            "sources": [{"source": "policy.pdf", "page": 1}],
            "found_context": True,
        },
    ):
        response = client.post("/chat", json={"question": "What is the return policy?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Returns are supported."


def test_ui_includes_phase1_token_testing_controls():
    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    for expected in (
        "Phase I Token Tester",
        "/auth/test",
        "/api/v1/hybrid-chat",
        "/api/v1/seller/hybrid-chat",
        "/api/v1/support/hybrid-chat",
        "/api/v1/operations/hybrid-chat",
        "/api/v1/admin/hybrid-chat",
    ):
        assert expected in response.text
