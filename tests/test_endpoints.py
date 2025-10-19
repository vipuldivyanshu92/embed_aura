"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/v1/healthz")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert "memory_backend" in data
    assert "slm_impl" in data


def test_hypothesize_returns_hypotheses(client: TestClient) -> None:
    """Test that hypothesize endpoint returns 2-3 hypotheses."""
    response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user",
            "input_text": "build an API",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "hypotheses" in data
    assert isinstance(data["hypotheses"], list)
    assert 2 <= len(data["hypotheses"]) <= 3

    # Check hypothesis structure
    for hyp in data["hypotheses"]:
        assert "id" in hyp
        assert "question" in hyp
        assert "rationale" in hyp
        assert "confidence" in hyp
        assert 0.0 <= hyp["confidence"] <= 1.0


def test_hypothesize_sorted_by_confidence(client: TestClient) -> None:
    """Test that hypotheses are sorted by confidence."""
    response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user2",
            "input_text": "fix a bug",
        },
    )

    assert response.status_code == 200
    data = response.json()

    confidences = [h["confidence"] for h in data["hypotheses"]]

    # Should be sorted descending
    assert confidences == sorted(confidences, reverse=True)


def test_hypothesize_auto_advance_flag(client: TestClient) -> None:
    """Test that auto_advance flag is present."""
    response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user3",
            "input_text": "some input",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "auto_advance" in data
    assert isinstance(data["auto_advance"], bool)


def test_execute_returns_enriched_prompt(client: TestClient) -> None:
    """Test that execute endpoint returns enriched prompt."""
    # First get hypotheses
    hyp_response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user4",
            "input_text": "create a web app",
        },
    )

    assert hyp_response.status_code == 200
    hypothesis_id = hyp_response.json()["hypotheses"][0]["id"]

    # Execute with selected hypothesis
    exec_response = client.post(
        "/v1/execute",
        json={
            "user_id": "test_user4",
            "input_text": "create a web app",
            "hypothesis_id": hypothesis_id,
        },
    )

    assert exec_response.status_code == 200
    data = exec_response.json()

    assert "enriched_prompt" in data
    assert "tokens_estimate" in data
    assert "context_breakdown" in data

    # Check enriched prompt is non-empty
    assert len(data["enriched_prompt"]) > 0

    # Check tokens estimate is reasonable
    assert data["tokens_estimate"] > 0


def test_execute_context_breakdown(client: TestClient) -> None:
    """Test that context breakdown is provided."""
    # Get hypotheses
    hyp_response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user5",
            "input_text": "optimize code",
        },
    )

    hypothesis_id = hyp_response.json()["hypotheses"][0]["id"]

    # Execute
    exec_response = client.post(
        "/v1/execute",
        json={
            "user_id": "test_user5",
            "input_text": "optimize code",
            "hypothesis_id": hypothesis_id,
        },
    )

    data = exec_response.json()
    breakdown = data["context_breakdown"]

    # Check that breakdown has expected sections
    expected_sections = [
        "goal_summary",
        "preferences_style",
        "critical_artifacts",
        "recent_history",
        "constraints",
        "task_specific_retrieval",
        "safety_system",
    ]

    for section in expected_sections:
        assert section in breakdown
        assert isinstance(breakdown[section], int)
        assert breakdown[section] >= 0

    # Total should be approximately token budget
    total = sum(breakdown.values())
    # Allow 20% variance as per spec
    assert total <= 6000 * 1.2


def test_memory_export(client: TestClient) -> None:
    """Test memory export endpoint."""
    # First create some interaction to generate memories
    client.post(
        "/v1/hypothesize",
        json={
            "user_id": "export_user",
            "input_text": "test input",
        },
    )

    # Export memories
    response = client.get("/v1/memory/export?user_id=export_user")

    assert response.status_code == 200
    data = response.json()

    assert "user_id" in data
    assert "memories" in data
    assert "persona" in data
    assert "exported_at" in data

    assert data["user_id"] == "export_user"


def test_memory_delete_local_backend(client: TestClient) -> None:
    """Test memory deletion (local backend only)."""
    user_id = "delete_user"

    # Create some data first
    client.post(
        "/v1/hypothesize",
        json={
            "user_id": user_id,
            "input_text": "test",
        },
    )

    # Delete all memories
    response = client.delete(f"/v1/memory?user_id={user_id}")

    assert response.status_code == 200
    data = response.json()

    assert "deleted_count" in data
    assert data["user_id"] == user_id


def test_hypothesize_empty_input_validation(client: TestClient) -> None:
    """Test that empty input is rejected."""
    response = client.post(
        "/v1/hypothesize",
        json={
            "user_id": "test_user",
            "input_text": "",
        },
    )

    # Should fail validation
    assert response.status_code == 422  # Validation error


def test_execute_invalid_hypothesis_id(client: TestClient) -> None:
    """Test execute with invalid hypothesis ID."""
    response = client.post(
        "/v1/execute",
        json={
            "user_id": "test_user",
            "input_text": "some input",
            "hypothesis_id": "invalid_id",
        },
    )

    # Should still complete but may not match any hypothesis
    # The endpoint should handle this gracefully
    assert response.status_code == 200
