
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


@router.get("")
def get_conversations(
    principal: UserPrincipal = Depends(get_current_principal),
):
    try:
        return {"sessions": list_sessions(principal)}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{session_id}/messages")
def get_conversation_messages(
    session_id: str,
    principal: UserPrincipal = Depends(get_current_principal),
):
    try:
        return get_session_history(principal, session_id)
    except PermissionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{session_id}")
def update_conversation_title(
    session_id: str,
    request: RenameSessionRequest,
    principal: UserPrincipal = Depends(get_current_principal),
):
    try:
        return rename_session(principal, session_id, request.title)
    except PermissionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/archive")
def archive_conversation(
    session_id: str,
    principal: UserPrincipal = Depends(get_current_principal),
):
    try:
        return archive_session(principal, session_id)
    except PermissionError as e:
        raise HTTPException(status_code=404, detail=str(e))
