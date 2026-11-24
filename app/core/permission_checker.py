
from fastapi import Depends, HTTPException, status

from app.core.auth import get_current_principal
from app.core.security import UserPrincipal
from app.core.role_policies import has_permission


def require_permission(permission: str):

    def permission_dependency(
        user: UserPrincipal = Depends(get_current_principal)
    ) -> UserPrincipal:

        if not has_permission(user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this feature."
            )

        return user

    return permission_dependency
