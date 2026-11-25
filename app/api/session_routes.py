
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import get_current_principal
from app.core.security import UserPrincipal

from app.orchestration.session_management import (
    list_sessions,
    get_session_history,
    rename_session,
    archive_session,
)

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversation Memory"])


class RenameSessionRequest(BaseModel):
    title: str


def _run_session_action(action_fn, *args, not_found_on_permission_error: bool = True, **kwargs):
    """Execute a session management operation with consistent HTTP error mapping."""
    try:
        return action_fn(*args, **kwargs)
    except PermissionError as e:
        status_code = 404 if not_found_on_permission_error else 403
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        print("Conversation session error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Conversation service is temporarily unavailable.",
        ) from e


@router.get("")
def get_conversations(
    principal: UserPrincipal = Depends(get_current_principal),
):
    return {
        "sessions": _run_session_action(
            list_sessions,
            principal,
            not_found_on_permission_error=False,
        )
    }


@router.get("/{session_id}/messages")
def get_conversation_messages(
    session_id: str,
    principal: UserPrincipal = Depends(get_current_principal),
):
    return _run_session_action(get_session_history, principal, session_id)


@router.patch("/{session_id}")
def update_conversation_title(
    session_id: str,
    request: RenameSessionRequest,
    principal: UserPrincipal = Depends(get_current_principal),
):
    return _run_session_action(rename_session, principal, session_id, request.title)


@router.post("/{session_id}/archive")
def archive_conversation(
    session_id: str,
    principal: UserPrincipal = Depends(get_current_principal),
):
    return _run_session_action(archive_session, principal, session_id)
