import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_session(client: AsyncClient):
    response = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "es"
    assert data["status"] == "active"
    assert "id" in data


async def test_get_session(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]
    token = create.json()["ws_token"]

    response = await client.get(
        f"/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == session_id


async def test_get_nonexistent_session(client: AsyncClient):
    # No session exists at this id, so no valid token exists either - the
    # request carries no Authorization header and the auth dependency now
    # rejects it (401) before the route body can 404 it. Same
    # anti-enumeration argument as the rest of this task: a missing session
    # and a bad token must be indistinguishable.
    response = await client.get("/sessions/99999")
    assert response.status_code == 401


async def test_complete_session(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]
    token = create.json()["ws_token"]

    response = await client.patch(
        f"/sessions/{session_id}/complete",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


async def test_delete_session(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]
    token = create.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert (
        await client.delete(f"/sessions/{session_id}", headers=headers)
    ).status_code == 204
    # The session is gone, so its (still-valid-looking) old token can no
    # longer be checked against anything - get_by_id returns None and
    # authorize() returns False before the route body ever runs. That's
    # indistinguishable from a wrong token by design, so this is 401, not
    # 404 (same anti-enumeration argument as the rest of this task).
    assert (
        await client.get(f"/sessions/{session_id}", headers=headers)
    ).status_code == 401


async def test_get_transcriptions_empty(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]
    token = create.json()["ws_token"]

    response = await client.get(
        f"/sessions/{session_id}/transcriptions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_get_translations_empty(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]
    token = create.json()["ws_token"]

    response = await client.get(
        f"/sessions/{session_id}/translations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_default_language_pair(client: AsyncClient):
    """Default pair is en->es (main use case)."""
    response = await client.post("/sessions/", json={})
    assert response.status_code == 201
    data = response.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "es"


async def test_create_session_returns_ws_token(client):
    response = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert response.status_code == 201
    token = response.json()["ws_token"]
    assert isinstance(token, str) and len(token) >= 32


async def test_get_session_never_leaks_ws_token(client):
    """Si el GET devolviera el token, el IDOR sólo se mudaría de endpoint."""
    created = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]

    fetched = await client.get(
        f"/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert fetched.status_code == 200
    assert "ws_token" not in fetched.json()


async def test_two_sessions_get_different_tokens(client):
    tokens = set()
    for _ in range(2):
        response = await client.post(
            "/sessions/", json={"source_language": "en", "target_language": "es"}
        )
        tokens.add(response.json()["ws_token"])
    assert len(tokens) == 2


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/sessions/{id}"),
        ("patch", "/sessions/{id}/complete"),
        ("delete", "/sessions/{id}"),
        ("get", "/sessions/{id}/transcriptions"),
        ("get", "/sessions/{id}/translations"),
    ],
)
async def test_session_routes_reject_foreign_token(client, method, path):
    """A real token, but for a different session, must not open this one -
    same property test_ws_rejects_foreign_session_token checks for the WS.
    Uses a valid-looking token (not just missing/malformed) so this can only
    pass if the dependency actually looks up *this* session_id's token."""
    a = (await client.post("/sessions/", json={})).json()
    b = (await client.post("/sessions/", json={})).json()

    response = await getattr(client, method)(
        path.format(id=b["id"]),
        headers={"Authorization": f"Bearer {a['ws_token']}"},
    )
    assert response.status_code == 401


async def test_unsupported_language_pair_rejected(client):
    response = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "de"}
    )
    assert response.status_code == 422


async def test_supported_language_pair_accepted(client):
    response = await client.post(
        "/sessions/", json={"source_language": "es", "target_language": "en"}
    )
    assert response.status_code == 201
