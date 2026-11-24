
import os
from fastapi import Header, HTTPException, status

from app.core.security import UserPrincipal, UserRole


def get_dev_current_user(
    x_dev_user_id: str = Header(default=None),
    x_dev_role: str = Header(default=None),
):
    # Development authentication must never run in production.
    environment = os.getenv("APP_ENV", "development").lower()

    if environment == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development authentication is disabled."
        )

    if not x_dev_user_id or not x_dev_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Development identity headers are required."
        )

    try:
        role = UserRole(x_dev_role.lower())
        return UserPrincipal(
            user_id=x_dev_user_id,
            role=role,
            is_active=True
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user role."
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user."
        )
