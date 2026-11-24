
from fastapi import Depends, HTTPException, status

from app.core.auth import get_current_principal
from app.core.security import UserPrincipal, UserRole


def require_roles(*allowed_roles: UserRole):

    def role_dependency(
        principal: UserPrincipal = Depends(get_current_principal)
    ) -> UserPrincipal:

        if principal.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not authorized to access this resource."
            )

        return principal

    return role_dependency
