"""
GPUGuard - Prometheus Metrics Exporter
Exposes GPU cluster metrics in Prometheus format on /metrics endpoint.
Includes job-level, node-level, and cluster-level metrics.
"""

from prometheus_client import (
    Gauge, Counter, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.gpu_job_simulator import GPUCluster, JobStatus

# ─── Prometheus Metrics ────────────────────────────────────────────────────────

REGISTRY = CollectorRegistry()

# Cluster-level
cluster_running_jobs = Gauge(
    "gpuguard_cluster_running_jobs_total",
    "Number of actively running training jobs",
    registry=REGISTRY,
)
cluster_failed_jobs = Gauge(
    "gpuguard_cluster_failed_jobs_total",
    "Number of currently failed training jobs",
    registry=REGISTRY,
)
cluster_completed_jobs_counter = Counter(
    "gpuguard_cluster_completed_jobs_total",
    "Total completed training jobs",
    registry=REGISTRY,
)
cluster_failure_events = Counter(
    "gpuguard_cluster_failure_events_total",
    "Total failure events observed",
    ["failure_type"],
    registry=REGISTRY,
)
cluster_restart_events = Counter(
    "gpuguard_cluster_restart_events_total",
    "Total auto-remediation restarts triggered",
    registry=REGISTRY,
)
cluster_avg_gpu_utilization = Gauge(
    "gpuguard_cluster_avg_gpu_utilization_pct",
    "Average GPU utilization across all running jobs",
    registry=REGISTRY,
)
cluster_total_throughput = Gauge(
    "gpuguard_cluster_total_throughput_tokens_per_sec",
    "Aggregate token throughput across all running jobs",
    registry=REGISTRY,
)

# Job-level
job_training_loss = Gauge(
    "gpuguard_job_training_loss",
    "Current training loss for a job",
    ["job_id", "model_name"],
    registry=REGISTRY,
)
job_training_step = Gauge(
    "gpuguard_job_training_step",
    "Current training step",
    ["job_id", "model_name"],
    registry=REGISTRY,
)
job_progress_pct = Gauge(
    "gpuguard_job_progress_pct",
    "Job completion percentage",
    ["job_id", "model_name"],
    registry=REGISTRY,
)
job_throughput = Gauge(
    "gpuguard_job_throughput_tokens_per_sec",
    "Training throughput in tokens/sec",
    ["job_id", "model_name"],
    registry=REGISTRY,
)
job_restart_count = Gauge(
    "gpuguard_job_restart_count",
    "Number of restarts for this job",
    ["job_id", "model_name"],
    registry=REGISTRY,
)

# Node-level
node_gpu_utilization = Gauge(
    "gpuguard_node_gpu_utilization_pct",
    "GPU utilization % for a node",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_gpu_memory_used_gb = Gauge(
    "gpuguard_node_gpu_memory_used_gb",
    "GPU memory used in GB",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_gpu_memory_utilization = Gauge(
    "gpuguard_node_gpu_memory_utilization_pct",
    "GPU memory utilization %",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_gpu_temp = Gauge(
    "gpuguard_node_gpu_temp_celsius",
    "GPU temperature in Celsius",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_nvlink_bandwidth = Gauge(
    "gpuguard_node_nvlink_bandwidth_gbps",
    "NVLink bandwidth in GB/s",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_healthy = Gauge(
    "gpuguard_node_healthy",
    "1 if node is healthy, 0 otherwise",
    ["node_id", "job_id"],
    registry=REGISTRY,
)
node_network_flap_count = Counter(
    "gpuguard_node_network_flap_total",
    "Total network flap events for a node",
    ["node_id", "job_id"],
    registry=REGISTRY,
)

# SLO metrics (populated by slo_engine)
slo_error_budget_remaining_pct = Gauge(
    "gpuguard_slo_error_budget_remaining_pct",
    "Remaining error budget percentage",
    ["slo_name"],
    registry=REGISTRY,
)
slo_burn_rate = Gauge(
    "gpuguard_slo_burn_rate",
    "Current error budget burn rate (1.0 = sustainable)",
    ["slo_name"],
    registry=REGISTRY,
)

# ─── Exporter ─────────────────────────────────────────────────────────────────

class MetricsExporter:
    def __init__(self, cluster: GPUCluster):
        self.cluster = cluster
        self._prev_failures = 0
        self._prev_restarts = 0
        self._prev_completed = 0

    def update(self):
        stats = self.cluster.get_cluster_stats()

        # Cluster metrics
        cluster_running_jobs.set(stats["running_jobs"])
        cluster_failed_jobs.set(stats["failed_jobs"])
        cluster_avg_gpu_utilization.set(round(stats["avg_gpu_utilization"], 2))
        cluster_total_throughput.set(round(stats["total_throughput_tokens_per_sec"], 0))

        # Counters (increment deltas)
        new_failures = stats["total_failures"] - self._prev_failures
        if new_failures > 0:
            cluster_failure_events.labels(failure_type="any").inc(new_failures)
            self._prev_failures = stats["total_failures"]

        new_restarts = stats["total_restarts"] - self._prev_restarts
        if new_restarts > 0:
            cluster_restart_events.inc(new_restarts)
            self._prev_restarts = stats["total_restarts"]

        new_completed = stats["completed_jobs"] - self._prev_completed
        if new_completed > 0:
            cluster_completed_jobs_counter.inc(new_completed)
            self._prev_completed = stats["completed_jobs"]

        # Per-job metrics
        for job in self.cluster.jobs.values():
            labels = {"job_id": job.job_id, "model_name": job.model_name}
            job_training_loss.labels(**labels).set(round(job.loss, 4))
            job_training_step.labels(**labels).set(job.step)
            job_progress_pct.labels(**labels).set(round(job.progress_pct, 2))
            job_throughput.labels(**labels).set(round(job.throughput_tokens_per_sec, 0))
            job_restart_count.labels(**labels).set(job.restart_count)

            # Per-node metrics
            for node in job.nodes:
                nlabels = {"node_id": node.node_id, "job_id": job.job_id}
                node_gpu_utilization.labels(**nlabels).set(round(node.gpu_utilization, 2))
                node_gpu_memory_used_gb.labels(**nlabels).set(round(node.gpu_memory_used_gb, 2))
                node_gpu_memory_utilization.labels(**nlabels).set(round(node.gpu_memory_utilization, 2))
                node_gpu_temp.labels(**nlabels).set(round(node.gpu_temp_celsius, 1))
                node_nvlink_bandwidth.labels(**nlabels).set(round(node.nvlink_bandwidth_gbps, 1))
                node_healthy.labels(**nlabels).set(1 if node.is_healthy else 0)


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="GPUGuard Metrics Exporter", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cluster = GPUCluster(max_concurrent_jobs=6)
exporter = MetricsExporter(cluster)

# Seed with initial jobs
for _ in range(3):
    cluster.spawn_job()


def _simulation_loop():
    """Background thread: advance simulation and update metrics every 5 seconds."""
    while True:
        cluster.tick()
        exporter.update()
        time.sleep(5)


sim_thread = threading.Thread(target=_simulation_loop, daemon=True)
sim_thread.start()


@app.get("/metrics", response_class=Response)
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "tick": cluster.tick_count}


@app.get("/cluster/stats")
def cluster_stats():
    return cluster.get_cluster_stats()


@app.get("/cluster/jobs")
def list_jobs():
    return [
        {
            "job_id": j.job_id,
            "model_name": j.model_name,
            "status": j.status.value,
            "step": j.step,
            "total_steps": j.total_steps,
            "progress_pct": round(j.progress_pct, 1),
            "loss": round(j.loss, 4),
            "throughput": round(j.throughput_tokens_per_sec, 0),
            "restart_count": j.restart_count,
            "num_nodes": j.num_nodes,
            "failure_type": j.failure_type.value,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "is_healthy": n.is_healthy,
                    "gpu_utilization": round(n.gpu_utilization, 1),
                    "gpu_memory_used_gb": round(n.gpu_memory_used_gb, 1),
                    "gpu_temp_celsius": round(n.gpu_temp_celsius, 1),
                }
                for n in j.nodes
            ],
        }
        for j in cluster.jobs.values()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
