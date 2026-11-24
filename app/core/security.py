
from dataclasses import dataclass
from enum import Enum


class UserRole(str, Enum):
    CUSTOMER = "customer"
    SELLER = "seller"
    SUPPORT = "support"
    OPERATIONS = "operations"
    ADMIN = "admin"


@dataclass(frozen=True)
class UserPrincipal:
    user_id: str
    role: UserRole
    is_active: bool

    def __post_init__(self):
        if not self.user_id or not self.user_id.strip():
            raise ValueError("Authenticated user ID is required.")

        if not isinstance(self.role, UserRole):
            raise ValueError("A valid user role is required.")

        if not self.is_active:
            raise PermissionError("User account is inactive.")


def require_roles(principal: UserPrincipal, allowed_roles):
    if not principal.is_active:
        raise PermissionError("Inactive user.")

    if principal.role not in allowed_roles:
        raise PermissionError(
            "You are not authorized to access this resource."
        )

    return True
