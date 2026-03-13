# Self-Learning Journal — NVIDIA AI Infrastructure Transition

**Candidate background:** 6+ years as a Software Engineer (backend/distributed systems)  
**Target role:** Systems Software Engineer, AI Infrastructure  
**Learning period:** ~3 weeks (this project was built in parallel)

---

## Why I built GPUGuard

The NVIDIA AI Infrastructure role is squarely in territory I haven't worked in professionally — GPU training infrastructure, HPC, SRE tooling for ML workloads. Rather than describing what I *would* build if given the chance, I built it.

This journal documents my learning process honestly: what I knew, what I had to learn, where I got stuck, and how I got unstuck. I believe showing the learning process is as important as showing the output.

---

## Week 1 — Understanding the domain

### What I already knew
- Distributed systems fundamentals (Raft consensus, CAP theorem, consistent hashing)
- Python async, REST APIs, Docker, basic Kubernetes
- CI/CD with GitLab
- Prometheus from monitoring a microservices backend

### What I didn't know and had to learn

**NCCL and collective communications**  
Starting point: I knew GPU training uses multiple GPUs, but not *how* they communicate.

Resources I used:
- NVIDIA NCCL documentation (developer.nvidia.com/nccl)
- "Efficient Large Scale Language Modeling with Megatron" (Narayanan et al., 2021)
- NCCL operational best practices blog post on NVIDIA developer blog

Key insight: `AllReduce` is the critical collective — gradients from all GPUs are summed and distributed back. A single slow node (straggler) blocks the entire ring. This is why `NCCL_COMM_TIMEOUT` is one of the most common failure modes in production — and why straggler detection matters so much that I built a CUSUM-based detector specifically for it.

**H100 GPU architecture**  
Resources used:
- NVIDIA H100 Tensor Core GPU Architecture whitepaper
- DCGM (Data Center GPU Manager) documentation for metric taxonomy

Key learning: NVLink 4.0 provides ~900 GB/s total bandwidth vs PCIe 5.0's ~128 GB/s. This is why `nvlink_bandwidth_gbps` is a first-class metric in the exporter — a drop here precedes NCCL failures by 20-60 seconds.

**Gradient checkpointing**  
Resources used:
- "Training Large Neural Networks" (Rajbhandari et al., ZeRO paper)
- PyTorch checkpoint documentation

Key learning: Checkpointing trades compute for memory, activations are recomputed during backward pass. This is why OOM failures can be mitigated by reducing micro-batch size rather than just restarting.

---

## Week 2 — SRE methodology for ML workloads

### Google SRE Book — what I read and why

I read chapters 3, 4, 5, and 13 specifically:

**Chapter 3 — Embracing Risk:** The concept of error budgets changed how I think about reliability. The key insight: 100% reliability is never the goal because reliability has a cost. An error budget is the agreed-upon unreliability you're allowed to spend.

*How I applied it:* The `SLOWindow` class implements exactly this — it tracks the rolling ratio of good/bad events, computes how much budget has been consumed, and calculates burn rate. I implemented Welford's online algorithm for rolling stats so the window doesn't require storing all historical values.

**Chapter 5 — Eliminating Toil:** "Toil is work that is manual, repetitive, automatable, tactical, devoid of enduring value." The auto-remediation engine is a direct translation of this — runbooks for every failure type, with a circuit breaker so the automation itself doesn't cause cascading failures.

**Chapter 13 — Emergency Response:** The incident log and MTTR tracking in `SLOEngine` came directly from this chapter. The key metric isn't just "did we fix it" but "how long did it take, and is that consistent?"

### Multi-window burn rate alerts

This took me a while to understand correctly. The alert rules I wrote use two windows:

```
# 2-minute window: catch fast burns (budget gone in <1 hour)
burn_rate > 14.4 for 2m → CRITICAL

# 5-minute window: catch sustained moderate burns
burn_rate > 6.0 for 5m → WARNING
```

The `14.4` threshold comes from: 1 hour / (1 - 0.995) = 200 hours of budget × 1/14.4 ≈ 1 hour to exhaustion. I had to work through the math several times before it clicked.

---

## Week 3 — Anomaly detection without a full ML stack

### Why statistical methods over ML models

The JD mentions observability tooling. I considered using an LSTM or Isolation Forest for anomaly detection, but decided against it:

1. A model needs training data — which you don't have for a new cluster
2. A model adds an inference dependency that can fail
3. Simpler methods (z-score, CUSUM) are more debuggable and explainable

**What I chose and why:**

*Z-score for spike detection:* Detects sudden deviations from the rolling mean. Catches loss spikes (potential gradient explosion) and memory spikes (pre-OOM warning). Used Welford's online algorithm so it's O(1) per update with no full-window storage.

*CUSUM for drift detection:* Detects gradual degradation that z-score misses. A slow node might only reduce throughput by 5% per tick — well within 3σ — but CUSUM accumulates the deviation and alarms after it sustains. This is the difference between catching a straggler at 30 seconds vs 5 minutes.

*IQR for outlier detection:* More robust than z-score for heavily skewed distributions (GPU util during startup, memory during checkpoint save). Uses quantiles rather than mean/std so outliers don't corrupt the baseline.

**What I got wrong first:**

My first CUSUM implementation reset on every alarm, which caused it to re-alarm immediately. The fix was to reset only after a genuine recovery (the circuit breaker pattern applied to anomaly detection). I caught this by writing the test `test_burn_rate_sustainable` which failed before the fix.

---

## Key things I would do differently in production

1. **Separate the simulation from the exporter.** In a real system, the exporter scrapes DCGM (NVIDIA's official GPU metrics daemon) instead of simulating. The interfaces are identical — swap `GPUCluster` for a DCGM gRPC client.

2. **Persistent incident store.** The incident log is in-memory. Production needs Postgres or a time-series DB.

3. **OpenTelemetry traces.** Prometheus metrics tell you *what* happened, traces tell you *where* in the call stack. For debugging NCCL timeouts, distributed tracing across nodes would be essential.

4. **Canary deployments.** The GitLab CI deploys directly to staging then manual approval to prod. In a real infrastructure team, you'd shadow-deploy and compare SLI metrics before cutover.

5. **Alert routing.** The Alertmanager config is minimal. Production needs routing by severity (critical → PagerDuty, warning → Slack), inhibition rules (don't page about node GPU temp if the node is already cordoned), and silencing during maintenance windows.

---

## Resources that were most valuable

| Resource | What it taught me |
|---|---|
| *Site Reliability Engineering* (Google, free online) | Error budgets, SLI/SLO/SLA, toil elimination |
| NVIDIA NCCL documentation | Collective communications, failure modes |
| DCGM documentation | GPU metric taxonomy (what to observe) |
| "Megatron-LM" paper (Narayanan 2021) | How LLM training is distributed across nodes |
| Prometheus best practices guide | Metric naming conventions, cardinality |
| "Designing Data-Intensive Applications" ch. 8 | Faults and partial failures in distributed systems |

---

## What I'm still learning

- Kubernetes operators (I understand deployments but not custom controllers)
- Terraform — the JD mentions CDK; I've used Pulumi but not CDK specifically
- CUDA programming model (I understand the GPU memory hierarchy conceptually but haven't written CUDA kernels)
- Slurm vs Kubernetes for HPC scheduling tradeoffs

I'm treating the last point as the first thing to dig into after submitting this application.

---

*Last updated: March 2026*
