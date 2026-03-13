"""
GPUGuard - Unit Tests
Tests for SLO engine, circuit breaker, and cluster simulation.
Demonstrates engineering rigor expected in a systems reliability role.
"""

import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from simulator.gpu_job_simulator import GPUCluster, TrainingJob, GPUNode, JobStatus, FailureType
from slo.slo_engine import SLOWindow, SLOEngine
from remediation.auto_remediation import AutoRemediationEngine, CircuitBreaker


# ─── SLO Window Tests ─────────────────────────────────────────────────────────

class TestSLOWindow:

    def test_perfect_availability(self):
        slo = SLOWindow("test", target_pct=99.5)
        for _ in range(100):
            slo.record(True)
        assert slo.current_sli_pct == 100.0
        assert slo.error_budget_remaining_pct == 100.0
        assert slo.burn_rate == 0.0
        assert slo.status() == "OK"

    def test_sli_calculation(self):
        slo = SLOWindow("test", target_pct=99.5)
        for _ in range(99):
            slo.record(True)
        slo.record(False)
        assert slo.current_sli_pct == pytest.approx(99.0, abs=0.01)

    def test_error_budget_consumption(self):
        slo = SLOWindow("test", target_pct=99.5)
        # 100 events, 2 bad = 98% SLI, target is 99.5% → budget = 0.5%
        # Actual bad rate = 2% → consumed > 100%
        for _ in range(98):
            slo.record(True)
        for _ in range(2):
            slo.record(False)
        assert slo.error_budget_remaining_pct == 0.0
        assert slo.error_budget_consumed_pct == 100.0

    def test_burn_rate_critical(self):
        slo = SLOWindow("test", target_pct=99.5)
        # Inject many failures to drive burn rate above 14.4x
        for _ in range(50):
            slo.record(False)
        assert slo.burn_rate > 14.4
        assert slo.status() == "CRITICAL"

    def test_burn_rate_sustainable(self):
        slo = SLOWindow("test", target_pct=99.5)
        for _ in range(995):
            slo.record(True)
        for _ in range(5):
            slo.record(False)
        # ~99.5% SLI = right at target, burn rate should be ~1
        assert 0.5 < slo.burn_rate < 2.0

    def test_rolling_window_eviction(self):
        slo = SLOWindow("test", target_pct=99.0)
        slo.window_ticks = 10  # shrink window for test
        # Fill with failures
        for _ in range(10):
            slo.record(False)
        assert len(slo.bad_ticks) == 10
        # Add more good events, old bads should be evicted
        for _ in range(10):
            slo.record(True)
        assert len(slo.bad_ticks) == 0
        assert len(slo.good_ticks) == 10

    def test_status_transitions(self):
        slo = SLOWindow("test", target_pct=99.5)
        # OK
        for _ in range(100):
            slo.record(True)
        assert slo.status() == "OK"
        # Inject failures to trigger WARNING
        for _ in range(40):
            slo.record(False)
        assert slo.status() in ("WARNING", "CRITICAL")


# ─── Circuit Breaker Tests ────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_allows_attempts_below_threshold(self):
        cb = CircuitBreaker(max_attempts=3, window_seconds=60, cooldown=120)
        assert cb.record_attempt("job-1") is True
        assert cb.record_attempt("job-1") is True
        assert not cb.is_open("job-1")

    def test_opens_after_max_attempts(self):
        cb = CircuitBreaker(max_attempts=3, window_seconds=60, cooldown=120)
        cb.record_attempt("job-1")
        cb.record_attempt("job-1")
        cb.record_attempt("job-1")  # This trips the breaker
        assert cb.is_open("job-1")
        assert cb.record_attempt("job-1") is False

    def test_different_keys_independent(self):
        cb = CircuitBreaker(max_attempts=2, window_seconds=60, cooldown=120)
        cb.record_attempt("job-1")
        cb.record_attempt("job-1")  # trips job-1
        assert cb.is_open("job-1")
        assert not cb.is_open("job-2")
        assert cb.record_attempt("job-2") is True

    def test_reset_clears_breaker(self):
        cb = CircuitBreaker(max_attempts=2, window_seconds=60, cooldown=120)
        cb.record_attempt("job-1")
        cb.record_attempt("job-1")
        assert cb.is_open("job-1")
        cb.reset("job-1")
        assert not cb.is_open("job-1")
        assert cb.record_attempt("job-1") is True


# ─── GPU Simulator Tests ──────────────────────────────────────────────────────

class TestGPUCluster:

    def test_job_spawning(self):
        cluster = GPUCluster(max_concurrent_jobs=3)
        job = cluster.spawn_job(model_name="test-model", num_nodes=2)
        assert job.job_id in cluster.jobs
        assert job.status == JobStatus.RUNNING
        assert len(job.nodes) == 2

    def test_loss_decreases_over_time(self):
        job = TrainingJob(total_steps=1000)
        losses = []
        for step in [0, 100, 500, 900]:
            job.step = step
            losses.append(job.simulate_loss())
        # Average trend should be decreasing (allowing for noise)
        assert losses[0] > losses[-1]

    def test_node_simulate_tick_running(self):
        node = GPUNode(node_id="test-node")
        node.simulate_tick(job_running=True)
        assert node.gpu_utilization > 0
        assert node.gpu_memory_used_gb > 0

    def test_node_simulate_tick_idle(self):
        node = GPUNode(node_id="test-node", gpu_utilization=90.0, gpu_memory_used_gb=60.0)
        node.simulate_tick(job_running=False)
        assert node.gpu_utilization < 90.0
        assert node.nvlink_bandwidth_gbps == 0.0

    def test_cluster_tick_advances_jobs(self):
        cluster = GPUCluster(max_concurrent_jobs=3)
        cluster.spawn_job()
        initial_steps = {jid: j.step for jid, j in cluster.jobs.items()}
        cluster.tick()
        for jid, job in cluster.jobs.items():
            if jid in initial_steps and job.status == JobStatus.RUNNING:
                assert job.step >= initial_steps[jid]

    def test_job_completion(self):
        cluster = GPUCluster(max_concurrent_jobs=1)
        job = cluster.spawn_job()
        job.total_steps = 5  # Fast completion
        job.step = 4
        cluster.tick()
        assert job.job_id not in cluster.jobs or cluster.jobs[job.job_id].status == JobStatus.COMPLETED
        assert len(cluster.completed_jobs) >= 1

    def test_failed_job_attempts_restart(self):
        cluster = GPUCluster(max_concurrent_jobs=2)
        job = cluster.spawn_job()
        job.status = JobStatus.FAILED
        initial_restarts = job.restart_count
        cluster.tick()
        if job.job_id in cluster.jobs:
            assert cluster.jobs[job.job_id].restart_count > initial_restarts or \
                   cluster.jobs[job.job_id].status == JobStatus.RECOVERING


# ─── Auto-Remediation Tests ───────────────────────────────────────────────────

class TestAutoRemediation:

    def test_oom_remediation(self):
        engine = AutoRemediationEngine()
        cluster = GPUCluster(max_concurrent_jobs=2)
        job = cluster.spawn_job()
        job.status = JobStatus.FAILED
        job.failure_type = FailureType.OOM
        job.step = 200
        job.checkpoint_step = 150

        slo_report = {"slos": {}}
        actions = engine.evaluate(cluster, slo_report)

        assert any(a["action"] == "restart_from_checkpoint" for a in [x.to_dict() for x in actions])

    def test_node_down_remediation(self):
        engine = AutoRemediationEngine()
        cluster = GPUCluster(max_concurrent_jobs=2)
        job = cluster.spawn_job(num_nodes=4)
        job.status = JobStatus.FAILED
        job.failure_type = FailureType.NODE_DOWN
        job.nodes[0].is_healthy = False
        job.checkpoint_step = 100
        job.step = 200

        slo_report = {"slos": {}}
        engine.evaluate(cluster, slo_report)
        assert engine.total_actions > 0

    def test_remediation_stats(self):
        engine = AutoRemediationEngine()
        assert engine.stats()["total_actions"] == 0
        assert engine.stats()["success_rate_pct"] == 100.0
