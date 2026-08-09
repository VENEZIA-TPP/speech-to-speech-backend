import secrets

from app.repositories.interfaces.translation_session_repository import (
    ITranslationSessionRepository,
)


async def authorize_session_token(
    session_repo: ITranslationSessionRepository,
    session_id: int,
    token: str | None,
) -> bool:
    """Constant-time check that `token` is this session's ws_token.

    Returns False for a missing session too, so a caller cannot distinguish
    "wrong token" from "no such session" — the same enumeration-safety
    property the WebSocket's 4401 close already has.
    """
    if not token:
        return False
    session = await session_repo.get_by_id(session_id)
    if session is None:
        return False
    return secrets.compare_digest(session.ws_token.encode(), token.encode())
