"""
GPUGuard - Anomaly Detection Engine
Lightweight statistical anomaly detection for GPU training metrics.
No ML framework required — uses rolling statistics (z-score, IQR, CUSUM)
so it runs embedded in the same process as the exporter.

Why this matters for NVIDIA AI Infrastructure:
  - Training loss spikes can indicate gradient explosions before a crash
  - Throughput drift indicates slow nodes 20-60s before NCCL timeout
  - Memory growth rate predicts OOM before it happens
  - Early detection enables graceful checkpoint-and-restart vs hard failure

Algorithms:
  1. Z-Score     — detects point anomalies (sudden spikes)
  2. CUSUM       — detects drift/trend anomalies (gradual degradation)
  3. IQR Fence   — robust outlier detection for noisy GPU metrics
"""

import math
import collections
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Deque

logger = logging.getLogger("anomaly_detection")


# ─── Rolling Statistics ───────────────────────────────────────────────────────

class RollingStats:
    """
    Welford's online algorithm for rolling mean and variance.
    O(1) per update, no need to store the full window.
    """
    def __init__(self, window: int = 60):
        self.window = window
        self._values: Deque[float] = collections.deque(maxlen=window)
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0  # sum of squared deviations

    def update(self, x: float):
        if len(self._values) == self.window:
            # Remove oldest value from running stats (approximate)
            old = self._values[0]
            old_mean = self._mean
            self._n -= 1
            if self._n > 0:
                self._mean = (self._mean * (self._n + 1) - old) / self._n
                self._M2 = max(0, self._M2 - (old - old_mean) * (old - self._mean))
            else:
                self._mean = 0.0
                self._M2 = 0.0

        self._values.append(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._M2 / self._n if self._n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def count(self) -> int:
        return self._n

    def z_score(self, x: float) -> float:
        if self.std == 0:
            return 0.0
        return (x - self.mean) / self.std


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart.
    Detects persistent shifts in the mean — ideal for slow throughput degradation.
    
    S+ accumulates upward deviations (useful for loss spikes, memory growth).
    S- accumulates downward deviations (useful for throughput drops).
    """
    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k = k    # allowance (usually 0.5 * sigma)
        self.h = h    # decision threshold
        self.S_pos = 0.0
        self.S_neg = 0.0

    def update(self, z: float) -> tuple[float, float]:
        """Takes z-score as input. Returns (S+, S-)."""
        self.S_pos = max(0.0, self.S_pos + z - self.k)
        self.S_neg = max(0.0, self.S_neg - z - self.k)
        return self.S_pos, self.S_neg

    def reset(self):
        self.S_pos = 0.0
        self.S_neg = 0.0

    @property
    def alarm_high(self) -> bool:
        return self.S_pos >= self.h

    @property
    def alarm_low(self) -> bool:
        return self.S_neg >= self.h


# ─── Anomaly Types ────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    metric: str
    entity_id: str          # job_id or node_id
    anomaly_type: str       # "spike", "drift_high", "drift_low", "outlier"
    value: float
    expected: float
    severity: str           # "warning", "critical"
    timestamp: float = field(default_factory=time.time)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "entity_id": self.entity_id,
            "anomaly_type": self.anomaly_type,
            "value": round(self.value, 4),
            "expected": round(self.expected, 4),
            "severity": self.severity,
            "timestamp": self.timestamp,
            "description": self.description,
        }


# ─── Per-Metric Tracker ───────────────────────────────────────────────────────

class MetricTracker:
    """
    Tracks one metric time series for one entity (job or node).
    Applies z-score, CUSUM, and IQR detection in parallel.
    Requires WARMUP_TICKS before raising any alarms to avoid cold-start false positives.
    """
    WARMUP_TICKS = 20
    Z_SCORE_THRESHOLD = 3.0     # 3-sigma rule
    Z_SCORE_CRITICAL = 4.5
    IQR_MULTIPLIER = 2.5        # less strict than standard 1.5 for noisy GPU metrics

    def __init__(self, metric_name: str, entity_id: str, window: int = 60):
        self.metric_name = metric_name
        self.entity_id = entity_id
        self.stats = RollingStats(window=window)
        self.cusum = CUSUMDetector(k=0.5, h=5.0)
        self._iqr_window: Deque[float] = collections.deque(maxlen=window)
        self._tick = 0

    def evaluate(self, value: float) -> list[AnomalyEvent]:
        self._tick += 1
        self.stats.update(value)
        self._iqr_window.append(value)
        anomalies = []

        if self._tick < self.WARMUP_TICKS:
            return anomalies

        z = self.stats.z_score(value)
        s_pos, s_neg = self.cusum.update(z)

        # 1. Z-Score spike detection
        if abs(z) >= self.Z_SCORE_THRESHOLD:
            severity = "critical" if abs(z) >= self.Z_SCORE_CRITICAL else "warning"
            direction = "above" if z > 0 else "below"
            anomalies.append(AnomalyEvent(
                metric=self.metric_name,
                entity_id=self.entity_id,
                anomaly_type="spike",
                value=value,
                expected=self.stats.mean,
                severity=severity,
                description=f"Value {direction} expected by {abs(z):.1f}σ (z={z:.2f})",
            ))

        # 2. CUSUM drift detection
        if self.cusum.alarm_high:
            anomalies.append(AnomalyEvent(
                metric=self.metric_name,
                entity_id=self.entity_id,
                anomaly_type="drift_high",
                value=value,
                expected=self.stats.mean,
                severity="warning",
                description=f"Sustained upward drift detected (CUSUM S+={s_pos:.2f})",
            ))
            self.cusum.reset()
        elif self.cusum.alarm_low:
            anomalies.append(AnomalyEvent(
                metric=self.metric_name,
                entity_id=self.entity_id,
                anomaly_type="drift_low",
                value=value,
                expected=self.stats.mean,
                severity="warning",
                description=f"Sustained downward drift detected (CUSUM S-={s_neg:.2f})",
            ))
            self.cusum.reset()

        # 3. IQR outlier detection (more robust for skewed distributions)
        if len(self._iqr_window) >= 20:
            sorted_vals = sorted(self._iqr_window)
            n = len(sorted_vals)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[(3 * n) // 4]
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - self.IQR_MULTIPLIER * iqr
                upper = q3 + self.IQR_MULTIPLIER * iqr
                if value < lower or value > upper:
                    # Only emit if not already caught by z-score (dedup)
                    if not any(a.anomaly_type == "spike" for a in anomalies):
                        anomalies.append(AnomalyEvent(
                            metric=self.metric_name,
                            entity_id=self.entity_id,
                            anomaly_type="outlier",
                            value=value,
                            expected=(q1 + q3) / 2,
                            severity="warning",
                            description=f"IQR outlier: value={value:.2f}, fence=[{lower:.2f}, {upper:.2f}]",
                        ))

        return anomalies


# ─── Main Engine ──────────────────────────────────────────────────────────────

class AnomalyDetectionEngine:
    """
    Manages per-job and per-node metric trackers.
    Called each simulation tick to evaluate all metrics.
    """

    TRACKED_JOB_METRICS = [
        ("loss", "training_loss"),
        ("throughput", "throughput_tokens_per_sec"),
    ]
    TRACKED_NODE_METRICS = [
        ("gpu_utilization", "gpu_utilization_pct"),
        ("gpu_memory_used_gb", "gpu_memory_used_gb"),
        ("gpu_temp_celsius", "gpu_temperature_celsius"),
    ]

    def __init__(self):
        self._job_trackers: dict[tuple, MetricTracker] = {}
        self._node_trackers: dict[tuple, MetricTracker] = {}
        self.anomaly_log: list[dict] = []
        self.total_anomalies = 0

    def _get_job_tracker(self, job_id: str, metric: str) -> MetricTracker:
        key = (job_id, metric)
        if key not in self._job_trackers:
            self._job_trackers[key] = MetricTracker(metric, job_id, window=60)
        return self._job_trackers[key]

    def _get_node_tracker(self, node_id: str, metric: str) -> MetricTracker:
        key = (node_id, metric)
        if key not in self._node_trackers:
            self._node_trackers[key] = MetricTracker(metric, node_id, window=60)
        return self._node_trackers[key]

    def evaluate(self, cluster) -> list[AnomalyEvent]:
        from simulator.gpu_job_simulator import JobStatus
        all_anomalies = []

        for job in cluster.jobs.values():
            if job.status != JobStatus.RUNNING:
                continue

            # Job-level metrics
            for attr, metric_name in self.TRACKED_JOB_METRICS:
                value = getattr(job, attr, None)
                if value is None:
                    continue
                tracker = self._get_job_tracker(job.job_id, metric_name)
                anomalies = tracker.evaluate(value)
                for a in anomalies:
                    logger.warning(f"[ANOMALY] {a.metric} on {a.entity_id}: {a.description}")
                    self.anomaly_log.append(a.to_dict())
                    self.total_anomalies += 1
                all_anomalies.extend(anomalies)

            # Node-level metrics
            for node in job.nodes:
                if not node.is_healthy:
                    continue
                for attr, metric_name in self.TRACKED_NODE_METRICS:
                    value = getattr(node, attr, None)
                    if value is None:
                        continue
                    tracker = self._get_node_tracker(node.node_id, metric_name)
                    anomalies = tracker.evaluate(value)
                    for a in anomalies:
                        logger.warning(f"[ANOMALY] {a.metric} on node {a.entity_id}: {a.description}")
                        self.anomaly_log.append(a.to_dict())
                        self.total_anomalies += 1
                    all_anomalies.extend(anomalies)

        # Evict trackers for jobs no longer in the cluster
        active_ids = {j.job_id for j in cluster.jobs.values()}
        stale_job_keys = [k for k in self._job_trackers if k[0] not in active_ids]
        for k in stale_job_keys:
            del self._job_trackers[k]

        return all_anomalies

    def stats(self) -> dict:
        recent = self.anomaly_log[-20:][::-1]
        by_type: dict[str, int] = {}
        for a in self.anomaly_log:
            by_type[a["anomaly_type"]] = by_type.get(a["anomaly_type"], 0) + 1
        return {
            "total_anomalies": self.total_anomalies,
            "by_type": by_type,
            "active_trackers": len(self._job_trackers) + len(self._node_trackers),
            "recent": recent,
        }
