"""
GPUGuard - GPU Training Job Simulator
Simulates a multi-node GPU training cluster with realistic failure scenarios.
Designed to generate observable metrics for SLO tracking and auto-remediation.
"""

import random
import time
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gpu_simulator")


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    RECOVERING = "recovering"


class FailureType(Enum):
    NONE = "none"
    OOM = "cuda_out_of_memory"
    NCCL_TIMEOUT = "nccl_comm_timeout"
    NODE_DOWN = "node_down"
    NETWORK_FLAP = "network_flap"
    CHECKSUM_MISMATCH = "gradient_checksum_mismatch"
    SLOW_NODE = "slow_node_straggler"


@dataclass
class GPUNode:
    node_id: str
    gpu_count: int = 8
    gpu_utilization: float = 0.0       # 0-100%
    gpu_memory_used_gb: float = 0.0    # GB used
    gpu_memory_total_gb: float = 80.0  # H100 80GB
    gpu_temp_celsius: float = 40.0
    nvlink_bandwidth_gbps: float = 0.0
    pcie_bandwidth_gbps: float = 0.0
    is_healthy: bool = True
    failure_type: FailureType = FailureType.NONE
    flap_count: int = 0

    @property
    def gpu_memory_utilization(self) -> float:
        return (self.gpu_memory_used_gb / self.gpu_memory_total_gb) * 100

    def simulate_tick(self, job_running: bool, failure_type: FailureType = FailureType.NONE):
        if not job_running or not self.is_healthy:
            self.gpu_utilization = max(0, self.gpu_utilization - random.uniform(5, 20))
            self.gpu_memory_used_gb = max(0, self.gpu_memory_used_gb - random.uniform(1, 5))
            self.nvlink_bandwidth_gbps = 0.0
            self.gpu_temp_celsius = max(35, self.gpu_temp_celsius - random.uniform(1, 3))
            return

        if failure_type == FailureType.OOM:
            self.gpu_memory_used_gb = min(self.gpu_memory_total_gb, self.gpu_memory_used_gb + random.uniform(2, 8))
            self.gpu_utilization = random.uniform(85, 95)
        elif failure_type == FailureType.SLOW_NODE:
            self.gpu_utilization = random.uniform(20, 40)  # straggler
            self.gpu_memory_used_gb = random.uniform(30, 50)
        else:
            # Normal training: high utilization with some jitter
            target_util = random.gauss(92, 3)
            self.gpu_utilization = max(0, min(100, target_util))
            self.gpu_memory_used_gb = random.uniform(55, 72)
            self.nvlink_bandwidth_gbps = random.gauss(450, 20)  # NVLink 4.0 ~900 GB/s total
            self.pcie_bandwidth_gbps = random.gauss(32, 2)
            self.gpu_temp_celsius = random.gauss(78, 3)


@dataclass
class TrainingJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = "llama-3-70b"
    num_nodes: int = 4
    status: JobStatus = JobStatus.QUEUED
    step: int = 0
    total_steps: int = 1000
    loss: float = 10.0
    throughput_tokens_per_sec: float = 0.0
    elapsed_seconds: float = 0.0
    failure_type: FailureType = FailureType.NONE
    restart_count: int = 0
    max_restarts: int = 3
    checkpoint_step: int = 0
    nodes: list = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        self.nodes = [
            GPUNode(node_id=f"node-{self.job_id}-{i}", gpu_count=8)
            for i in range(self.num_nodes)
        ]

    @property
    def progress_pct(self) -> float:
        return (self.step / self.total_steps) * 100 if self.total_steps > 0 else 0

    def simulate_loss(self) -> float:
        """Realistic loss curve: exponential decay with noise."""
        base = 10.0 * math.exp(-0.004 * self.step)
        noise = random.gauss(0, 0.05)
        spike = 0.3 if random.random() < 0.02 else 0  # occasional loss spike
        return max(0.1, base + noise + spike)

    def simulate_throughput(self) -> float:
        """Tokens/sec depends on node health and collective efficiency."""
        healthy_nodes = sum(1 for n in self.nodes if n.is_healthy)
        base = 125_000 * (self.num_nodes * 8)  # tokens/sec per H100
        efficiency = 0.85 if healthy_nodes == self.num_nodes else 0.4
        jitter = random.gauss(1.0, 0.03)
        return base * efficiency * jitter if healthy_nodes > 0 else 0


class GPUCluster:
    """
    Simulates a GPU cluster running multiple concurrent training jobs.
    Injects realistic failure scenarios on a probabilistic schedule.
    """

    FAILURE_RATES = {
        FailureType.NETWORK_FLAP: 0.002,
        FailureType.NCCL_TIMEOUT: 0.001,
        FailureType.OOM: 0.0008,
        FailureType.NODE_DOWN: 0.0005,
        FailureType.SLOW_NODE: 0.003,
    }

    def __init__(self, max_concurrent_jobs: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: dict[str, TrainingJob] = {}
        self.completed_jobs: list[TrainingJob] = []
        self.total_failures = 0
        self.total_restarts = 0
        self.tick_count = 0

    def spawn_job(self, model_name: str = "llama-3-70b", num_nodes: int = 4) -> TrainingJob:
        job = TrainingJob(model_name=model_name, num_nodes=num_nodes, total_steps=random.randint(500, 2000))
        job.status = JobStatus.RUNNING
        self.jobs[job.job_id] = job
        logger.info(f"Spawned job {job.job_id} ({model_name}, {num_nodes} nodes)")
        return job

    def _inject_failure(self, job: TrainingJob) -> Optional[FailureType]:
        for failure_type, rate in self.FAILURE_RATES.items():
            if random.random() < rate:
                return failure_type
        return None

    def _handle_failure(self, job: TrainingJob, failure: FailureType):
        self.total_failures += 1
        logger.warning(f"Job {job.job_id} encountered {failure.value} at step {job.step}")
        job.failure_type = failure

        if failure == FailureType.NODE_DOWN:
            # Kill a random node
            victim = random.choice(job.nodes)
            victim.is_healthy = False
            job.status = JobStatus.FAILED
        elif failure == FailureType.NCCL_TIMEOUT:
            job.status = JobStatus.FAILED
        elif failure == FailureType.NETWORK_FLAP:
            for node in job.nodes:
                node.flap_count += 1
            # Network flap may not kill the job, just degrade it
            if random.random() < 0.3:
                job.status = JobStatus.FAILED
        elif failure == FailureType.OOM:
            job.status = JobStatus.FAILED
        elif failure == FailureType.SLOW_NODE:
            slow_node = random.choice(job.nodes)
            slow_node.failure_type = FailureType.SLOW_NODE

    def _attempt_restart(self, job: TrainingJob) -> bool:
        """Auto-remediation: restart from last checkpoint."""
        if job.restart_count >= job.max_restarts:
            logger.error(f"Job {job.job_id} exceeded max restarts ({job.max_restarts}). Marking terminal.")
            return False

        job.restart_count += 1
        self.total_restarts += 1
        rollback_steps = random.randint(20, 100)
        job.step = max(0, job.checkpoint_step)
        job.failure_type = FailureType.NONE

        # Heal nodes (simulating node replacement / network recovery)
        for node in job.nodes:
            node.is_healthy = True
            node.failure_type = FailureType.NONE
            node.flap_count = 0

        job.status = JobStatus.RECOVERING
        logger.info(f"Job {job.job_id} restarting (attempt {job.restart_count}) from step {job.step}")
        return True

    def tick(self):
        """Advance simulation by one time step (~30 seconds of simulated time)."""
        self.tick_count += 1

        # Spawn new jobs if cluster has capacity
        if len(self.jobs) < self.max_concurrent_jobs and random.random() < 0.15:
            models = ["llama-3-70b", "mistral-8x7b", "falcon-180b", "gemma-27b"]
            self.spawn_job(model_name=random.choice(models), num_nodes=random.choice([2, 4, 8]))

        for job_id, job in list(self.jobs.items()):
            if job.status == JobStatus.RECOVERING:
                # Give recovering jobs a tick to warm up
                job.status = JobStatus.RUNNING
                continue

            if job.status == JobStatus.RUNNING:
                # Advance step
                job.step += random.randint(1, 5)
                job.elapsed_seconds += 30
                job.loss = job.simulate_loss()
                job.throughput_tokens_per_sec = job.simulate_throughput()

                # Save checkpoint every 50 steps
                if job.step % 50 < 5:
                    job.checkpoint_step = job.step

                # Tick all nodes
                for node in job.nodes:
                    node.simulate_tick(job_running=True, failure_type=node.failure_type)

                # Inject potential failure
                failure = self._inject_failure(job)
                if failure:
                    self._handle_failure(job, failure)

                # Check completion
                if job.step >= job.total_steps:
                    job.status = JobStatus.COMPLETED
                    logger.info(f"Job {job.job_id} completed at step {job.step}")
                    self.completed_jobs.append(job)
                    del self.jobs[job_id]

            elif job.status == JobStatus.FAILED:
                restarted = self._attempt_restart(job)
                if not restarted:
                    self.completed_jobs.append(job)
                    del self.jobs[job_id]

    def get_cluster_stats(self) -> dict:
        running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]
        failed = [j for j in self.jobs.values() if j.status == JobStatus.FAILED]
        return {
            "total_jobs": len(self.jobs),
            "running_jobs": len(running),
            "failed_jobs": len(failed),
            "completed_jobs": len(self.completed_jobs),
            "total_failures": self.total_failures,
            "total_restarts": self.total_restarts,
            "avg_gpu_utilization": (
                sum(n.gpu_utilization for j in running for n in j.nodes) /
                max(1, sum(len(j.nodes) for j in running))
            ),
            "total_throughput_tokens_per_sec": sum(j.throughput_tokens_per_sec for j in running),
        }
