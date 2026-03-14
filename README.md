# GPUGuard 🛡️
### AI Training Infrastructure Reliability & Observability Platform

> Built as a self-directed portfolio project to demonstrate systems engineering, SRE, and AI infrastructure skills

---

## What it does

GPUGuard simulates a production-grade GPU training cluster (multi-node H100s running LLM training jobs) and wraps it with the **exact reliability tooling** the NVIDIA AI Infrastructure team builds and maintains:

| Capability | Implementation |
|---|---|
| GPU cluster simulation | Multi-node training jobs with realistic failure modes (OOM, NCCL timeout, node down, network flap, straggler) |
| Metrics & observability | Prometheus exporter with job/node/cluster-level metrics |
| SLO tracking | Rolling-window SLI/SLO with error budget and burn rate (Google SRE Book methodology) |
| Auto-remediation | Runbook-driven recovery engine with circuit breaker (prevents flapping) |
| Alerting | Prometheus alert rules with multi-window burn rate (2m + 5m windows) |
| CI/CD | Full GitLab pipeline: lint → test → build → security scan → staged deploy |
| Containerization | Docker Compose with Prometheus + Grafana + Alertmanager |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GPUGuard Platform                       │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │  GPU Cluster  │   │  SLO Engine  │   │  Auto-Remediation│ │
│  │  Simulator    │──▶│  Error Budget│──▶│  Circuit Breaker │ │
│  │  (5s ticks)   │   │  Burn Rate   │   │  Runbooks        │ │
│  └──────┬────────┘   └──────────────┘   └─────────────────┘ │
│         │                                                     │
│  ┌──────▼────────────────────────────────────────────────┐  │
│  │           FastAPI  (REST + /metrics)                   │  │
│  └──────┬────────────────────────────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  Prometheus   │────▶│   Grafana    │     │ Alertmanager  │
   │  (scrapes 5s) │     │  Dashboards  │     │ (PD/Slack)    │
   └──────────────┘     └──────────────┘     └──────────────┘
```

---

## SLOs Defined

| SLO | Target | What it measures |
|---|---|---|
| `job_availability` | 99.5% | % of ticks with < 5% of jobs in FAILED state |
| `training_throughput` | 95.0% | % of ticks total token/s > 80% of baseline |
| `gpu_utilization` | 90.0% | % of ticks cluster avg GPU util > 85% |
| `mttr` | 90.0% | % of failures recovered in < 5 minutes |

Burn rate thresholds (per Google SRE book):
- `> 14.4x` → **CRITICAL** — page immediately, budget gone in <1 hour
- `> 6.0x`  → **WARNING** — investigate, budget gone in ~2.5 hours
- `< 10% remaining` → **AT_RISK** — freeze non-critical experiments

---

## Failure Scenarios & Runbooks

| Failure | Probability/tick | Auto-Remediation |
|---|---|---|
| Network flap | 0.2% | Exponential backoff retry, reset flap counters |
| Slow node (straggler) | 0.3% | Identify straggler, restart with re-profiling |
| NCCL comm timeout | 0.1% | Reset NCCL communicators, restart from checkpoint |
| CUDA OOM | 0.08% | Reduce micro-batch size, restart from checkpoint |
| Node down | 0.05% | Cordon node, reschedule on healthy nodes |
| Gradient checksum | 0.03% | Roll back 100 steps, reload stable checkpoint |

All remediations are protected by a **Circuit Breaker** (3 attempts / 5-minute window) to prevent cascading retry storms.

---

## Quick Start

### Option A: Docker Compose (recommended)
```bash
git clone https://github.com/yourhandle/gpuguard
cd gpuguard
docker compose up -d

# Services:
# GPUGuard API  →  http://localhost:8000
# Prometheus    →  http://localhost:9090
# Grafana       →  http://localhost:3000  (admin/gpuguard)
# Alertmanager  →  http://localhost:9093
```

### Option B: Local Python
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && uvicorn api.main:app --reload --port 8000
```

### API Endpoints
```
GET  /metrics               # Prometheus scrape endpoint
GET  /api/v1/cluster        # Cluster-level stats
GET  /api/v1/jobs           # All jobs with per-node metrics
GET  /api/v1/slo            # SLO report with error budgets
GET  /api/v1/incidents      # Incident log and MTTR stats
GET  /api/v1/remediation    # Auto-remediation action log
POST /api/v1/jobs/spawn     # Spawn a new training job
```

---

## Running Tests
```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Key Prometheus Queries

```promql
# SLO burn rate — alert if > 14.4
gpuguard_slo_burn_rate{slo_name="job_availability"}

# Remaining error budget per SLO
gpuguard_slo_error_budget_remaining_pct

# Average cluster GPU utilization
gpuguard_cluster_avg_gpu_utilization_pct

# Job throughput (tokens/sec) per model
sum by (model_name) (gpuguard_job_throughput_tokens_per_sec)

# Node health heatmap
gpuguard_node_healthy

# Rate of failure events (last 5m)
rate(gpuguard_cluster_failure_events_total[5m])

# MTTR trend
gpuguard_slo_error_budget_remaining_pct{slo_name="mttr"}
```


---

## Tech Stack
- **Python 3.12** — core simulation, SLO engine, remediation
- **FastAPI** — REST API + Prometheus endpoint
- **Prometheus** — metrics collection + alerting
- **Grafana** — dashboards
- **Docker Compose** — local infra-as-code
- **GitLab CI** — complete pipeline
- **pytest** — unit + integration tests

---

## Self-Learning Notes

This project was built while learning:
- Google SRE Book (error budgets, burn rates, SLI/SLO/SLA)
- Prometheus data model and PromQL
- NCCL architecture and GPU collective communications failure modes
- Grafana dashboard design for GPU training workloads
- FastAPI async patterns and middleware

**Resources used:**
- *Site Reliability Engineering* (Google) — chapters 3, 4, 5, 13
- NVIDIA NCCL documentation
- Prometheus best practices guide
- DCGM (Data Center GPU Manager) metric taxonomy

---

*Maintained by: Jaspreet
