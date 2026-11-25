from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import auth_provider
from app.core.embeddings import embed_query
from main import app


def test_head_root_returns_200():
    client = TestClient(app)
    response = client.head("/")
    assert response.status_code == 200


def test_embed_query_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        embed_query("   ")


def test_embed_query_uses_hf_api_and_falls_back_to_onnx(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setattr(
        "app.core.embeddings._embed_via_hf_api",
        lambda text, token: (_ for _ in ()).throw(RuntimeError("HF down")),
    )
    monkeypatch.setattr(
        "app.core.embeddings._embed_via_onnx",
        lambda text: [0.5] * 384,
    )

    result = embed_query("What is the return policy?")
    assert len(result) == 384
    assert result[0] == 0.5


def test_embed_via_onnx_normalization_with_mock_session(monkeypatch):
    from app.core import embeddings

    fake_encoding = MagicMock()
    fake_encoding.ids = [101, 2054, 102]
    fake_encoding.attention_mask = [1, 1, 1]
    fake_encoding.type_ids = [0, 0, 0]

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = fake_encoding

    fake_session = MagicMock()
    fake_session.run.return_value = [np.ones((1, 3, 384), dtype=np.float32)]

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "app.core.embeddings._load_onnx_components",
        lambda: (fake_session, fake_tokenizer),
    )

    vec = embeddings.embed_query("return policy")
    assert len(vec) == 384
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_phase1_auth_provider_caches_verified_token(monkeypatch):
    monkeypatch.setenv("PHASE1_API_BASE_URL", "https://auth.example.com")
    monkeypatch.setenv("AUTH_CACHE_TTL_SECONDS", "60")
    auth_provider._AUTH_CACHE.clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "user_id": "USR-1",
        "role": "customer",
        "is_active": True,
    }

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    monkeypatch.setattr(auth_provider, "_get_auth_client", lambda: mock_client)

    provider = auth_provider.PhaseOneAuthenticationProvider()
    first = provider.authenticate("jwt-token-123")
    second = provider.authenticate("jwt-token-123")

    assert first == second
    assert mock_client.get.call_count == 1
