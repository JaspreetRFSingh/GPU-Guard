"""
GPUGuard - Integration Tests
Tests the full API stack end-to-end using ASGI test client.
Validates that endpoints return correct schemas and that the
simulation + SLO + remediation pipeline is wired up correctly.
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Spin up the full FastAPI app and return a TestClient."""
    from api.main import app
    # Give the background simulation thread a few ticks to warm up
    with TestClient(app) as c:
        time.sleep(2.0)
        yield c


class TestHealthEndpoints:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "tick" in body
        assert body["tick"] >= 0

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        body = r.text
        # Must contain our custom metrics
        assert "gpuguard_cluster_running_jobs_total" in body
        assert "gpuguard_cluster_avg_gpu_utilization_pct" in body
        assert "gpuguard_slo_error_budget_remaining_pct" in body

    def test_metrics_has_required_labels(self, client):
        r = client.get("/metrics")
        body = r.text
        # Node metrics must have node_id and job_id labels
        lines_with_node = [l for l in body.splitlines() if "gpuguard_node_gpu_utilization_pct" in l and "{" in l]
        if lines_with_node:
            assert "node_id=" in lines_with_node[0]
            assert "job_id=" in lines_with_node[0]


class TestClusterEndpoints:

    def test_cluster_stats_schema(self, client):
        r = client.get("/api/v1/cluster")
        assert r.status_code == 200
        body = r.json()
        required_keys = ["total_jobs", "running_jobs", "failed_jobs",
                         "completed_jobs", "total_failures", "total_restarts",
                         "avg_gpu_utilization", "total_throughput_tokens_per_sec"]
        for key in required_keys:
            assert key in body, f"Missing key: {key}"

    def test_cluster_jobs_within_capacity(self, client):
        r = client.get("/api/v1/cluster")
        body = r.json()
        assert body["total_jobs"] <= 6  # max_concurrent_jobs

    def test_jobs_endpoint_returns_list(self, client):
        r = client.get("/api/v1/jobs")
        assert r.status_code == 200
        jobs = r.json()
        assert isinstance(jobs, list)

    def test_job_schema_complete(self, client):
        r = client.get("/api/v1/jobs")
        jobs = r.json()
        if not jobs:
            pytest.skip("No active jobs in cluster")
        job = jobs[0]
        required = ["job_id", "model_name", "status", "step", "total_steps",
                    "progress_pct", "loss", "throughput_tokens_per_sec",
                    "restart_count", "num_nodes", "failure_type", "nodes"]
        for key in required:
            assert key in job, f"Job missing key: {key}"

    def test_job_nodes_schema(self, client):
        r = client.get("/api/v1/jobs")
        jobs = r.json()
        for job in jobs:
            for node in job["nodes"]:
                assert "node_id" in node
                assert "is_healthy" in node
                assert "gpu_utilization" in node
                assert "gpu_memory_used_gb" in node
                assert "gpu_temp_celsius" in node
                assert 0 <= node["gpu_utilization"] <= 100

    def test_progress_pct_in_range(self, client):
        r = client.get("/api/v1/jobs")
        jobs = r.json()
        for job in jobs:
            assert 0 <= job["progress_pct"] <= 100

    def test_spawn_job_endpoint(self, client):
        r = client.post("/api/v1/jobs/spawn", params={"model_name": "test-model", "num_nodes": 2})
        # Could be 200 (spawned) or 429 (capacity)
        assert r.status_code in (200, 429)
        if r.status_code == 200:
            body = r.json()
            assert "job_id" in body
            assert body["model_name"] == "test-model"
            assert body["num_nodes"] == 2


class TestSLOEndpoints:

    def test_slo_report_schema(self, client):
        r = client.get("/api/v1/slo")
        assert r.status_code == 200
        body = r.json()
        assert "overall_status" in body
        assert "slos" in body
        assert body["overall_status"] in ("OK", "WARNING", "AT_RISK", "CRITICAL")

    def test_slo_names_present(self, client):
        r = client.get("/api/v1/slo")
        slos = r.json()["slos"]
        expected = {"job_availability", "training_throughput", "gpu_utilization", "mttr"}
        # At least some should be present after warmup
        assert len(set(slos.keys()) & expected) > 0

    def test_slo_values_in_range(self, client):
        r = client.get("/api/v1/slo")
        for name, slo in r.json()["slos"].items():
            assert 0 <= slo["current_sli_pct"] <= 100, f"{name}: SLI out of range"
            assert 0 <= slo["error_budget_remaining_pct"] <= 100, f"{name}: budget out of range"
            assert slo["burn_rate"] >= 0, f"{name}: burn_rate negative"

    def test_slo_status_valid(self, client):
        r = client.get("/api/v1/slo")
        valid_statuses = {"OK", "WARNING", "AT_RISK", "CRITICAL"}
        for name, slo in r.json()["slos"].items():
            assert slo["status"] in valid_statuses, f"{name} has invalid status: {slo['status']}"


class TestRemediationEndpoints:

    def test_remediation_stats_schema(self, client):
        r = client.get("/api/v1/remediation")
        assert r.status_code == 200
        body = r.json()
        assert "total_actions" in body
        assert "successful_actions" in body
        assert "success_rate_pct" in body

    def test_success_rate_in_range(self, client):
        r = client.get("/api/v1/remediation")
        body = r.json()
        assert 0 <= body["success_rate_pct"] <= 100


class TestIncidentEndpoints:

    def test_incidents_schema(self, client):
        r = client.get("/api/v1/incidents")
        assert r.status_code == 200
        body = r.json()
        assert "active_failures" in body
        assert "resolved_failures" in body
        assert "recent_incidents" in body
        assert isinstance(body["recent_incidents"], list)

    def test_active_failures_non_negative(self, client):
        r = client.get("/api/v1/incidents")
        assert r.json()["active_failures"] >= 0


class TestPrometheusMetricValues:
    """
    Validates that Prometheus metric values are within physically valid ranges.
    This catches simulation bugs that would produce garbage data in dashboards.
    """

    def _parse_metric(self, text: str, metric_name: str) -> list[float]:
        values = []
        for line in text.splitlines():
            if line.startswith(metric_name) and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[-1]))
                    except ValueError:
                        pass
        return values

    def test_gpu_utilization_range(self, client):
        text = client.get("/metrics").text
        vals = self._parse_metric(text, "gpuguard_node_gpu_utilization_pct")
        for v in vals:
            assert 0 <= v <= 100, f"GPU util out of range: {v}"

    def test_gpu_memory_range(self, client):
        text = client.get("/metrics").text
        vals = self._parse_metric(text, "gpuguard_node_gpu_memory_used_gb")
        for v in vals:
            assert 0 <= v <= 80, f"GPU memory out of range: {v}"  # H100 = 80GB

    def test_throughput_non_negative(self, client):
        text = client.get("/metrics").text
        vals = self._parse_metric(text, "gpuguard_cluster_total_throughput_tokens_per_sec")
        for v in vals:
            assert v >= 0, f"Throughput negative: {v}"

    def test_error_budget_range(self, client):
        text = client.get("/metrics").text
        vals = self._parse_metric(text, "gpuguard_slo_error_budget_remaining_pct")
        for v in vals:
            assert 0 <= v <= 100, f"Error budget out of range: {v}"

    def test_node_healthy_binary(self, client):
        text = client.get("/metrics").text
        vals = self._parse_metric(text, "gpuguard_node_healthy")
        for v in vals:
            assert v in (0.0, 1.0), f"node_healthy not binary: {v}"
