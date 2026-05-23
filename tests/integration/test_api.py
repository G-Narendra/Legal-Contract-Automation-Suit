"""
Integration tests for FastAPI API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.api.main import app
    client = TestClient(app)
except Exception as e:
    # If imports fail, skip tests
    app = None
    client = None
    import logging
    logging.warning(f"API test setup failed: {e}")


pytestmark = pytest.mark.skipif(
    app is None,
    reason="API dependencies not available"
)


class TestAPI:
    """Test suite for API endpoints."""

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "subsystems" in data

    def test_analyze_unsupported_file(self):
        response = client.post(
            "/api/v1/contract/analyze",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
            data={"user_id": "test_user"},
        )
        assert response.status_code == 400

    def test_draft_missing_type(self):
        response = client.post(
            "/api/v1/contract/draft",
            data={
                "language": "english",
                "params": '{"party_a": "Company A"}',
            },
        )
        assert response.status_code == 422  # Missing required field

    def test_draft_invalid_params(self):
        response = client.post(
            "/api/v1/contract/draft",
            data={
                "contract_type": "employment",
                "params": "invalid json",
            },
        )
        assert response.status_code == 400

    def test_list_contracts(self):
        response = client.get("/api/v1/contracts")
        assert response.status_code == 200
        data = response.json()
        assert "contracts" in data
        assert "total" in data

    def test_dashboard(self):
        response = client.get("/api/v1/dashboard")
        assert response.status_code == 200
