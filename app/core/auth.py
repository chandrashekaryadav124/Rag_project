
import os

from fastapi import Request, HTTPException, status

from app.core.security import UserPrincipal
from app.core.dev_auth import get_dev_current_user
from app.core.auth_provider import PhaseOneAuthenticationProvider


def get_current_principal(request: Request) -> UserPrincipal:

    environment = os.getenv("APP_ENV", "development").lower()
    provider_name = os.getenv("AUTH_PROVIDER", "development").lower()

    is_render = os.getenv("RENDER", "").lower() in ("true", "1")

    # Never allow development authentication in production.
    if (environment == "production" or is_render) and provider_name != "phase1":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production authentication is not configured."
        )

    # Local development authentication.
    if provider_name == "development":

        try:
            return get_dev_current_user(
                x_dev_user_id=request.headers.get("X-Dev-User-Id"),
                x_dev_role=request.headers.get("X-Dev-Role"),
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid development user role."
            )
        except PermissionError:
            raise HTTPException(
                status_code=403,
                detail="Inactive development user."
            )

    # Real Phase I authentication.
    if provider_name == "phase1":

        authorization = request.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")

        if not separator or scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="A valid Bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            provider = PhaseOneAuthenticationProvider()
            return provider.authenticate(token.strip())

        except PermissionError:
            raise HTTPException(
                status_code=403,
                detail="User account is inactive or unauthorized."
            )

        except ValueError:
            raise HTTPException(
                status_code=401,
                detail="Phase I authentication rejected the token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Phase I authentication service is unavailable."
            )

    raise HTTPException(
        status_code=500,
        detail="Invalid authentication provider configuration."
    )
