
from datetime import datetime, timezone

from app.core.security import UserPrincipal
from app.core.supabase_client import get_supabase_client


def _owner(principal: UserPrincipal) -> dict:
    if not principal.is_active:
        raise PermissionError("Inactive user.")

    return {
        "owner_user_id": principal.user_id,
        "owner_role": principal.role.value.upper(),
    }


def get_or_create_session(
    principal: UserPrincipal,
    session_id: str | None = None,
) -> dict:
    owner = _owner(principal)
    client = get_supabase_client()

    if session_id:
        result = (
            client.table("conversation_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("owner_user_id", owner["owner_user_id"])
            .eq("owner_role", owner["owner_role"])
            .eq("is_archived", False)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise PermissionError(
                "Conversation session not found or not owned by this user."
            )

        return result.data[0]

    result = (
        client.table("conversation_sessions")
        .insert({
            **owner,
            "title": "New conversation",
            "is_archived": False,
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError("Failed to create conversation session.")

    return result.data[0]


def save_message(
    principal: UserPrincipal,
    session_id: str,
    message_type: str,
    content: str,
) -> dict:
    if message_type not in {"user", "assistant", "system", "tool"}:
        raise ValueError("Invalid conversation message type.")

    if not content or not content.strip():
        raise ValueError("Message content cannot be empty.")

    owner = _owner(principal)
    client = get_supabase_client()

    result = (
        client.table("conversation_messages")
        .insert({
            **owner,
            "session_id": session_id,
            "message_type": message_type,
            "content": content.strip(),
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError("Failed to save conversation message.")

    now = datetime.now(timezone.utc).isoformat()

    (
        client.table("conversation_sessions")
        .update({
            "updated_at": now,
            "last_message_at": now,
        })
        .eq("id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .execute()
    )

    return result.data[0]


def get_recent_messages(
    principal: UserPrincipal,
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    if limit < 1 or limit > 50:
        raise ValueError("Message history limit must be between 1 and 50.")

    owner = _owner(principal)
    client = get_supabase_client()

    result = (
        client.table("conversation_messages")
        .select("message_type, content, created_at")
        .eq("session_id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return list(reversed(result.data or []))
