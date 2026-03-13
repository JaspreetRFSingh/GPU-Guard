"""
GPUGuard - Main API
Central FastAPI application integrating simulator, SLO engine, and auto-remediation.
Serves the dashboard frontend and exposes REST + Prometheus endpoints.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

from simulator.gpu_job_simulator import GPUCluster, JobStatus
from slo.slo_engine import SLOEngine
from remediation.auto_remediation import AutoRemediationEngine
from exporter.metrics_exporter import MetricsExporter, REGISTRY

app = FastAPI(
    title="GPUGuard — AI Training Infrastructure Reliability Platform",
    description="Observability, SLO tracking, and auto-remediation for GPU training clusters",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── Core Components ──────────────────────────────────────────────────────────
cluster = GPUCluster(max_concurrent_jobs=6)
slo_engine = SLOEngine()
remediation_engine = AutoRemediationEngine()
metrics_exporter = MetricsExporter(cluster)

# Seed initial jobs
for _ in range(3):
    cluster.spawn_job()

_latest_slo_report: dict = {}
_latest_remediation_actions: list = []
_tick_count = 0


def _main_loop():
    """Background loop: simulate → evaluate SLOs → auto-remediate → export metrics."""
    global _latest_slo_report, _latest_remediation_actions, _tick_count
    while True:
        cluster.tick()
        report = slo_engine.evaluate(cluster)
        actions = remediation_engine.evaluate(cluster, report)
        metrics_exporter.update()

        _latest_slo_report = report
        _latest_remediation_actions = [a.to_dict() for a in actions] if actions else []
        _tick_count += 1
        time.sleep(5)


sim_thread = threading.Thread(target=_main_loop, daemon=True)
sim_thread.start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/metrics", response_class=Response, tags=["Observability"])
def prometheus_metrics():
    """Prometheus scrape endpoint. Add to prometheus.yml as a scrape target."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "tick": _tick_count, "uptime_seconds": _tick_count * 5}


@app.get("/api/v1/cluster", tags=["Cluster"])
def get_cluster():
    return cluster.get_cluster_stats()


@app.get("/api/v1/jobs", tags=["Jobs"])
def list_jobs():
    result = []
    for j in cluster.jobs.values():
        result.append({
            "job_id": j.job_id,
            "model_name": j.model_name,
            "status": j.status.value,
            "step": j.step,
            "total_steps": j.total_steps,
            "progress_pct": round(j.progress_pct, 1),
            "loss": round(j.loss, 4),
            "throughput_tokens_per_sec": round(j.throughput_tokens_per_sec, 0),
            "restart_count": j.restart_count,
            "num_nodes": j.num_nodes,
            "failure_type": j.failure_type.value,
            "elapsed_seconds": round(j.elapsed_seconds, 0),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "is_healthy": n.is_healthy,
                    "gpu_utilization": round(n.gpu_utilization, 1),
                    "gpu_memory_used_gb": round(n.gpu_memory_used_gb, 1),
                    "gpu_memory_utilization_pct": round(n.gpu_memory_utilization, 1),
                    "gpu_temp_celsius": round(n.gpu_temp_celsius, 1),
                    "nvlink_bandwidth_gbps": round(n.nvlink_bandwidth_gbps, 1),
                    "flap_count": n.flap_count,
                }
                for n in j.nodes
            ],
        })
    return result


@app.get("/api/v1/slo", tags=["SLO"])
def get_slo_report():
    return _latest_slo_report


@app.get("/api/v1/remediation", tags=["Remediation"])
def get_remediation_stats():
    return {
        **remediation_engine.stats(),
        "latest_actions": _latest_remediation_actions,
    }


@app.get("/api/v1/incidents", tags=["Incidents"])
def get_incidents():
    return {
        "active_failures": len(slo_engine.active_failures),
        "resolved_failures": len(slo_engine.resolved_failures),
        "recent_incidents": slo_engine.incident_log[-20:][::-1],
        "recent_mttr_avg_seconds": (
            round(
                sum(e.mttr_seconds for e in slo_engine.resolved_failures[-10:]) /
                max(1, len(slo_engine.resolved_failures[-10:])),
                1,
            ) if slo_engine.resolved_failures else None
        ),
    }


@app.post("/api/v1/jobs/spawn", tags=["Jobs"])
def spawn_job(model_name: str = "llama-3-70b", num_nodes: int = 4):
    if len(cluster.jobs) >= cluster.max_concurrent_jobs:
        raise HTTPException(status_code=429, detail="Cluster at capacity")
    job = cluster.spawn_job(model_name=model_name, num_nodes=num_nodes)
    return {"job_id": job.job_id, "model_name": model_name, "num_nodes": num_nodes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
