import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.fastapi_server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_docs_endpoint():
    response = client.get("/docs")
    assert response.status_code == 200

def test_recommendations_endpoint_structure():
    response = client.get("/api/v1/recommendations")
    assert response.status_code in [200, 422]  # 422 for missing required params

@pytest.mark.asyncio
async def test_api_startup():
    assert app is not None
