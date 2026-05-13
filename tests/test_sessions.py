import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_session(client: AsyncClient):
    response = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
    assert response.status_code == 201
    data = response.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "es"
    assert data["status"] == "active"
    assert "id" in data


async def test_get_session(client: AsyncClient):
    create = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
    session_id = create.json()["id"]

    response = await client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["id"] == session_id


async def test_get_nonexistent_session(client: AsyncClient):
    response = await client.get("/sessions/99999")
    assert response.status_code == 404


async def test_complete_session(client: AsyncClient):
    create = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
    session_id = create.json()["id"]

    response = await client.patch(f"/sessions/{session_id}/complete")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


async def test_delete_session(client: AsyncClient):
    create = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
    session_id = create.json()["id"]

    assert (await client.delete(f"/sessions/{session_id}")).status_code == 204
    assert (await client.get(f"/sessions/{session_id}")).status_code == 404


async def test_get_transcriptions_empty(client: AsyncClient):
    create = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
    session_id = create.json()["id"]

    response = await client.get(f"/sessions/{session_id}/transcriptions")
    assert response.status_code == 200
    assert response.json() == []


async def test_get_translations_empty(client: AsyncClient):
    create = await client.post("/sessions/", json={"source_language": "en", "target_language": "es"})
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
