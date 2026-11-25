
import hashlib
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.core.security import UserPrincipal, UserRole


_AUTH_CLIENT: Optional[httpx.Client] = None
_AUTH_CACHE_LOCK = threading.Lock()
_AUTH_CACHE: dict[str, tuple[float, UserPrincipal]] = {}
_MAX_CACHE_ENTRIES = 512


def _get_auth_client() -> httpx.Client:
    global _AUTH_CLIENT
    if _AUTH_CLIENT is None:
        _AUTH_CLIENT = httpx.Client(timeout=15.0)
    return _AUTH_CLIENT


def _cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("AUTH_CACHE_TTL_SECONDS", "60")))
    except ValueError:
        return 60.0


def _token_cache_key(base_url: str, token: str) -> str:
    return hashlib.sha256(f"{base_url}|{token}".encode("utf-8")).hexdigest()


class AuthenticationProvider(ABC):

    @abstractmethod
    def authenticate(self, token: str) -> UserPrincipal:
        """Validate credentials and return a verified user."""
        raise NotImplementedError


class DevelopmentAuthenticationProvider(AuthenticationProvider):

    def authenticate(self, token: str) -> UserPrincipal:
        raise NotImplementedError(
            "Development authentication uses request headers, "
            "not bearer tokens."
        )


class PhaseOneAuthenticationProvider(AuthenticationProvider):

    def authenticate(self, token: str) -> UserPrincipal:
        """
        Validate the JWT through the Phase I authentication endpoint.
        Never decode or trust the JWT locally.
        """

        base_url = os.getenv(
            "PHASE1_API_BASE_URL",
            ""
        ).rstrip("/")

        if not base_url:
            raise RuntimeError(
                "PHASE1_API_BASE_URL is not configured."
            )

        url = f"{base_url}/api/v1/auth/me"
        ttl = _cache_ttl_seconds()
        cache_key = _token_cache_key(base_url, token)

        if ttl > 0:
            now = time.monotonic()
            with _AUTH_CACHE_LOCK:
                cached = _AUTH_CACHE.get(cache_key)
                if cached and cached[0] > now:
                    return cached[1]
                if cached:
                    _AUTH_CACHE.pop(cache_key, None)

        try:
            response = _get_auth_client().get(
                url,
                headers={
                    "Authorization": f"Bearer {token}"
                },
            )

            if response.status_code == 401:
                raise ValueError("Phase I rejected the token.")

            if response.status_code == 403:
                raise PermissionError("User account is not authorized.")

            response.raise_for_status()

            user_data = response.json()

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to Phase I authentication service."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Phase I authentication service returned an error."
            ) from exc

        if not isinstance(user_data, dict):
            raise ValueError("Invalid Phase I user response.")

        user_id = user_data.get("user_id")
        role_value = user_data.get("role")
        is_active = user_data.get("is_active")

        if not user_id or not role_value or is_active is None:
            raise ValueError(
                "Phase I response is missing required user fields."
            )

        try:
            role = UserRole(str(role_value).lower())
        except ValueError as exc:
            raise ValueError(
                "Phase I returned an unsupported user role."
            ) from exc

        if is_active is not True:
            raise PermissionError("User account is inactive.")

        principal = UserPrincipal(
            user_id=str(user_id),
            role=role,
            is_active=True
        )

        if ttl > 0:
            with _AUTH_CACHE_LOCK:
                if len(_AUTH_CACHE) >= _MAX_CACHE_ENTRIES:
                    _AUTH_CACHE.clear()
                _AUTH_CACHE[cache_key] = (time.monotonic() + ttl, principal)

        return principal
