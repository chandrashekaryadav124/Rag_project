
import os
import httpx
from abc import ABC, abstractmethod

from app.core.security import UserPrincipal, UserRole


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

        try:
            response = httpx.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}"
                },
                timeout=15.0
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

        return UserPrincipal(
            user_id=str(user_id),
            role=role,
            is_active=True
        )
