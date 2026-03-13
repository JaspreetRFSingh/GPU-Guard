"""
GPUGuard - SLO Engine
Implements Google SRE-style SLOs with error budget tracking and burn rate alerts.
Directly mirrors what the NVIDIA AI Infrastructure team tracks for training clusters.

SLOs defined:
  1. Job Availability     — % of time jobs are not in FAILED state (target: 99.5%)
  2. Training Throughput  — % of ticks throughput > 80% of baseline (target: 95%)
  3. MTTR                 — Mean Time To Recovery after failure < 5 minutes (target: 90%)
  4. GPU Utilization      — % of time cluster GPU util > 85% (target: 90%)
"""

import time
import collections
from dataclasses import dataclass, field
from typing import Deque
import logging

logger = logging.getLogger("slo_engine")

WINDOW_SECONDS = 3600       # 1-hour rolling window
TICK_INTERVAL_SECONDS = 5   # matches simulation loop


@dataclass
class SLOWindow:
    """Rolling window of good/bad events for a single SLO."""
    name: str
    target_pct: float           # e.g. 99.5 means 99.5%
    window_ticks: int = field(init=False)
    good_ticks: Deque[int] = field(default_factory=collections.deque)
    bad_ticks: Deque[int] = field(default_factory=collections.deque)
    total_events: int = 0
    _tick_counter: int = 0

    def __post_init__(self):
        self.window_ticks = WINDOW_SECONDS // TICK_INTERVAL_SECONDS  # 720

    def record(self, is_good: bool):
        self._tick_counter += 1
        self.total_events += 1

        if is_good:
            self.good_ticks.append(self._tick_counter)
        else:
            self.bad_ticks.append(self._tick_counter)

        # Evict events outside rolling window
        cutoff = self._tick_counter - self.window_ticks
        while self.good_ticks and self.good_ticks[0] <= cutoff:
            self.good_ticks.popleft()
        while self.bad_ticks and self.bad_ticks[0] <= cutoff:
            self.bad_ticks.popleft()

    @property
    def current_window_events(self) -> int:
        return len(self.good_ticks) + len(self.bad_ticks)

    @property
    def current_sli_pct(self) -> float:
        """Current SLI: good events / total events in window (%)."""
        total = self.current_window_events
        if total == 0:
            return 100.0
        return (len(self.good_ticks) / total) * 100

    @property
    def error_budget_total_pct(self) -> float:
        """Total error budget = 100% - target%."""
        return 100.0 - self.target_pct

    @property
    def error_budget_consumed_pct(self) -> float:
        """How much of the error budget has been consumed."""
        if self.error_budget_total_pct == 0:
            return 100.0
        deficit = max(0, self.target_pct - self.current_sli_pct)
        return min(100.0, (deficit / self.error_budget_total_pct) * 100)

    @property
    def error_budget_remaining_pct(self) -> float:
        return max(0.0, 100.0 - self.error_budget_consumed_pct)

    @property
    def burn_rate(self) -> float:
        """
        Burn rate > 1 means budget is being consumed faster than it replenishes.
        Burn rate > 14.4 = critical (budget gone in <1 hour if sustained).
        """
        bad = len(self.bad_ticks)
        total = self.current_window_events
        if total == 0:
            return 0.0
        actual_error_rate = bad / total
        allowed_error_rate = self.error_budget_total_pct / 100.0
        if allowed_error_rate == 0:
            return float("inf")
        return actual_error_rate / allowed_error_rate

    def status(self) -> str:
        if self.burn_rate >= 14.4:
            return "CRITICAL"
        elif self.burn_rate >= 6.0:
            return "WARNING"
        elif self.error_budget_remaining_pct < 10:
            return "AT_RISK"
        return "OK"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_pct": self.target_pct,
            "current_sli_pct": round(self.current_sli_pct, 3),
            "error_budget_remaining_pct": round(self.error_budget_remaining_pct, 2),
            "burn_rate": round(self.burn_rate, 3),
            "status": self.status(),
            "window_events": self.current_window_events,
            "good_events": len(self.good_ticks),
            "bad_events": len(self.bad_ticks),
        }


@dataclass
class FailureEvent:
    job_id: str
    failure_type: str
    occurred_at: float
    resolved_at: float = 0.0

    @property
    def resolved(self) -> bool:
        return self.resolved_at > 0

    @property
    def mttr_seconds(self) -> float:
        if not self.resolved:
            return 0.0
        return self.resolved_at - self.occurred_at


class SLOEngine:
    """
    Evaluates SLOs every tick against the live cluster state.
    Provides error budget reports and multi-window burn rate alerts.
    """

    THROUGHPUT_BASELINE_TOKENS_PER_SEC = 500_000  # tokens/sec per 4-node job baseline
    THROUGHPUT_THRESHOLD_PCT = 0.80
    GPU_UTIL_THRESHOLD_PCT = 85.0
    MTTR_TARGET_SECONDS = 300.0   # 5 minutes

    def __init__(self):
        self.job_availability = SLOWindow("job_availability", target_pct=99.5)
        self.training_throughput = SLOWindow("training_throughput", target_pct=95.0)
        self.gpu_utilization = SLOWindow("gpu_utilization", target_pct=90.0)
        self.mttr = SLOWindow("mttr", target_pct=90.0)

        self.active_failures: dict[str, FailureEvent] = {}
        self.resolved_failures: list[FailureEvent] = []
        self.incident_log: list[dict] = []

    def evaluate(self, cluster) -> dict:
        """Called every simulation tick. Evaluates all SLOs and returns report."""
        from simulator.gpu_job_simulator import JobStatus, FailureType

        jobs = list(cluster.jobs.values())
        running = [j for j in jobs if j.status == JobStatus.RUNNING]
        failed = [j for j in jobs if j.status == JobStatus.FAILED]

        # ── SLO 1: Job Availability ───────────────────────────────────────────
        if jobs:
            availability_good = len(failed) == 0 or (len(failed) / len(jobs)) < 0.05
            self.job_availability.record(availability_good)

        # ── SLO 2: Training Throughput ────────────────────────────────────────
        if running:
            total_throughput = sum(j.throughput_tokens_per_sec for j in running)
            expected_throughput = len(running) * self.THROUGHPUT_BASELINE_TOKENS_PER_SEC
            throughput_ok = total_throughput >= expected_throughput * self.THROUGHPUT_THRESHOLD_PCT
            self.training_throughput.record(throughput_ok)

        # ── SLO 3: GPU Utilization ────────────────────────────────────────────
        all_nodes = [n for j in running for n in j.nodes if n.is_healthy]
        if all_nodes:
            avg_util = sum(n.gpu_utilization for n in all_nodes) / len(all_nodes)
            self.gpu_utilization.record(avg_util >= self.GPU_UTIL_THRESHOLD_PCT)

        # ── SLO 4: MTTR ───────────────────────────────────────────────────────
        now = time.time()
        for job in failed:
            if job.job_id not in self.active_failures:
                self.active_failures[job.job_id] = FailureEvent(
                    job_id=job.job_id,
                    failure_type=job.failure_type.value,
                    occurred_at=now,
                )
                self._log_incident("FAILURE_DETECTED", job.job_id, job.failure_type.value)

        # Resolve failures (job recovered or restarted)
        for job_id in list(self.active_failures.keys()):
            active_job_ids = {j.job_id for j in jobs}
            if job_id not in active_job_ids:
                # Job completed or was deleted — resolved
                event = self.active_failures.pop(job_id)
                event.resolved_at = now
                self.resolved_failures.append(event)
                mttr_ok = event.mttr_seconds <= self.MTTR_TARGET_SECONDS
                self.mttr.record(mttr_ok)
                self._log_incident("FAILURE_RESOLVED", job_id, event.failure_type,
                                   extra={"mttr_seconds": round(event.mttr_seconds, 1)})

        return self.report()

    def _log_incident(self, event_type: str, job_id: str, failure_type: str, extra: dict = None):
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "job_id": job_id,
            "failure_type": failure_type,
            **(extra or {}),
        }
        self.incident_log.append(entry)
        logger.info(f"[INCIDENT] {event_type} | job={job_id} | failure={failure_type}")

    def report(self) -> dict:
        slos = [
            self.job_availability.to_dict(),
            self.training_throughput.to_dict(),
            self.gpu_utilization.to_dict(),
            self.mttr.to_dict(),
        ]
        overall = "OK"
        for slo in slos:
            if slo["status"] == "CRITICAL":
                overall = "CRITICAL"
                break
            elif slo["status"] in ("WARNING", "AT_RISK") and overall == "OK":
                overall = slo["status"]

        recent_mttr = None
        if self.resolved_failures:
            recent = self.resolved_failures[-5:]
            recent_mttr = round(sum(e.mttr_seconds for e in recent) / len(recent), 1)

        return {
            "overall_status": overall,
            "slos": {s["name"]: s for s in slos},
            "active_failures": len(self.active_failures),
            "recent_mttr_avg_seconds": recent_mttr,
            "recent_incidents": self.incident_log[-10:][::-1],
        }
