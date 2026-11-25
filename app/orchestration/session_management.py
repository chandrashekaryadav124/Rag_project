
from datetime import datetime, timezone

from app.core.security import UserPrincipal
from app.core.supabase_client import get_supabase_client


def _owner(principal: UserPrincipal):
    if not principal.is_active:
        raise PermissionError("Inactive user")

    return {
        "owner_user_id": principal.user_id,
        "owner_role": principal.role.value.upper(),
    }


def list_sessions(principal: UserPrincipal):
    owner = _owner(principal)
    client = get_supabase_client()

    result = (
        client.table("conversation_sessions")
        .select("*")
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .eq("is_archived", False)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def get_session_history(principal: UserPrincipal, session_id: str):
    owner = _owner(principal)
    client = get_supabase_client()

    session = (
        client.table("conversation_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .limit(1)
        .execute()
    )

    if not session.data:
        raise PermissionError("Conversation not found or access denied")

    messages = (
        client.table("conversation_messages")
        .select("*")
        .eq("session_id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .order("created_at", desc=False)
        .execute()
    )

    return {
        "session": session.data[0],
        "messages": messages.data or [],
    }


def rename_session(
    principal: UserPrincipal,
    session_id: str,
    title: str,
):
    owner = _owner(principal)

    if not title or not title.strip():
        raise ValueError("Conversation title cannot be empty")

    client = get_supabase_client()

    result = (
        client.table("conversation_sessions")
        .update({
            "title": title.strip()[:150],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .execute()
    )

    if not result.data:
        raise PermissionError("Conversation not found or access denied")

    return result.data[0]


def archive_session(principal: UserPrincipal, session_id: str):
    owner = _owner(principal)
    client = get_supabase_client()

    result = (
        client.table("conversation_sessions")
        .update({
            "is_archived": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", session_id)
        .eq("owner_user_id", owner["owner_user_id"])
        .eq("owner_role", owner["owner_role"])
        .eq("is_archived", False)
        .execute()
    )

    if not result.data:
        raise PermissionError("Conversation not found or already archived")

    return {"message": "Conversation archived successfully"}
