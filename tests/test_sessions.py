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

    response = await client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["id"] == session_id


async def test_get_nonexistent_session(client: AsyncClient):
    response = await client.get("/sessions/99999")
    assert response.status_code == 404


async def test_complete_session(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]

    response = await client.patch(f"/sessions/{session_id}/complete")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


async def test_delete_session(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]

    assert (await client.delete(f"/sessions/{session_id}")).status_code == 204
    assert (await client.get(f"/sessions/{session_id}")).status_code == 404


async def test_get_transcriptions_empty(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]

    response = await client.get(f"/sessions/{session_id}/transcriptions")
    assert response.status_code == 200
    assert response.json() == []


async def test_get_translations_empty(client: AsyncClient):
    create = await client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = create.json()["id"]

    response = await client.get(f"/sessions/{session_id}/translations")
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

    fetched = await client.get(f"/sessions/{session_id}")
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
