#!/usr/bin/env python3
"""
GPUGuard CLI
Interactive command-line dashboard for the GPU training cluster simulator.
Uses only stdlib + optional 'rich' for pretty output.

Usage:
  python cli.py                    # Run live dashboard
  python cli.py --ticks 100        # Run 100 ticks then exit
  python cli.py --inject oom       # Inject a specific failure type
  python cli.py --export-metrics   # Print Prometheus text format to stdout
"""

import sys
import os
import time
import argparse
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from simulator.gpu_job_simulator import GPUCluster, JobStatus, FailureType
from slo.slo_engine import SLOEngine
from remediation.auto_remediation import AutoRemediationEngine
from anomaly.anomaly_detection import AnomalyDetectionEngine

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import BarColumn, Progress, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


def _plain_bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    filled_count = round(value / 100 * width)
    return filled * filled_count + empty * (width - filled_count)


def _status_icon(status: str) -> str:
    return {"running": "▶", "failed": "✗", "recovering": "↻", "completed": "✓"}.get(status, "?")


def _slo_color(status: str) -> str:
    return {"OK": "green", "WARN": "yellow", "CRIT": "red", "RISK": "yellow"}.get(status, "white")


def render_rich(cluster, slo_engine, remediation_engine, anomaly_engine, tick: int):
    slo_report = slo_engine.report()
    remediation_stats = remediation_engine.stats()
    anomaly_stats = anomaly_engine.stats()

    layout = Layout()

    # Header
    overall = slo_report.get("overall_status", "OK")
    color = {"OK": "green", "WARNING": "yellow", "CRITICAL": "red"}.get(overall, "white")
    header = Text(f"  ⬡ GPUGuard — AI Training Infrastructure  [tick {tick}]", style="bold")
    status_badge = Text(f" {overall} ", style=f"bold {color} on {color}")

    # Cluster metrics
    stats = cluster.get_cluster_stats()
    metrics_table = Table.grid(expand=True, padding=(0, 2))
    metrics_table.add_column(justify="left")
    metrics_table.add_column(justify="left")
    metrics_table.add_column(justify="left")
    metrics_table.add_column(justify="left")
    metrics_table.add_row(
        f"[cyan]Jobs running:[/cyan] [bold]{stats['running_jobs']}[/bold]",
        f"[cyan]GPU util:[/cyan] [bold]{stats['avg_gpu_utilization']:.1f}%[/bold]",
        f"[cyan]Throughput:[/cyan] [bold]{stats['total_throughput_tokens_per_sec']/1000:.0f}K tok/s[/bold]",
        f"[cyan]Remediations:[/cyan] [bold]{remediation_stats['total_actions']}[/bold] "
        f"({remediation_stats['success_rate_pct']}% ok)",
    )

    # SLO table
    slo_table = Table(title="SLO Error Budgets", show_header=True, header_style="bold cyan",
                      show_edge=True, padding=(0, 1))
    slo_table.add_column("SLO", min_width=22)
    slo_table.add_column("Target", justify="right", min_width=8)
    slo_table.add_column("SLI", justify="right", min_width=8)
    slo_table.add_column("Budget Left", min_width=24)
    slo_table.add_column("Burn Rate", justify="right", min_width=10)
    slo_table.add_column("Status", justify="center", min_width=8)

    for name, slo in slo_report.get("slos", {}).items():
        budget = slo["error_budget_remaining_pct"]
        br = slo["burn_rate"]
        st = slo["status"]
        color = _slo_color(st)
        bar = _plain_bar(budget, width=16)
        slo_table.add_row(
            name.replace("_", " "),
            f"{slo['target_pct']}%",
            f"{slo['current_sli_pct']:.2f}%",
            f"[{color}]{bar}[/{color}] {budget:.1f}%",
            f"[{color}]{br:.2f}×[/{color}]",
            f"[bold {color}]{st}[/bold {color}]",
        )

    # Jobs table
    jobs_table = Table(title="Active Jobs", show_header=True, header_style="bold cyan",
                       show_edge=True, padding=(0, 1))
    jobs_table.add_column("Job ID", min_width=8)
    jobs_table.add_column("Model", min_width=16)
    jobs_table.add_column("Progress", min_width=25)
    jobs_table.add_column("Loss", justify="right", min_width=8)
    jobs_table.add_column("Tok/s", justify="right", min_width=10)
    jobs_table.add_column("↺", justify="center", min_width=4)
    jobs_table.add_column("Status", min_width=12)

    for job in cluster.jobs.values():
        pct = job.progress_pct
        bar = _plain_bar(pct, width=14)
        st_color = {"running": "green", "failed": "red", "recovering": "yellow"}.get(job.status.value, "white")
        icon = _status_icon(job.status.value)
        jobs_table.add_row(
            f"[bold]{job.job_id}[/bold]",
            job.model_name,
            f"[green]{bar}[/green] {pct:.0f}%",
            f"{job.loss:.4f}",
            f"{job.throughput_tokens_per_sec/1000:.0f}K",
            str(job.restart_count),
            f"[{st_color}]{icon} {job.status.value}[/{st_color}]",
        )

    # Anomaly log
    anomaly_table = Table(title=f"Anomalies [{anomaly_stats['total_anomalies']} total]",
                          show_header=True, header_style="bold cyan",
                          show_edge=True, padding=(0, 1))
    anomaly_table.add_column("Metric", min_width=28)
    anomaly_table.add_column("Entity", min_width=12)
    anomaly_table.add_column("Type", min_width=12)
    anomaly_table.add_column("Value", justify="right", min_width=10)
    anomaly_table.add_column("Severity", min_width=10)

    for a in anomaly_stats.get("recent", [])[:5]:
        sev_color = "red" if a["severity"] == "critical" else "yellow"
        anomaly_table.add_row(
            a["metric"], a["entity_id"][:12], a["anomaly_type"],
            f"{a['value']:.3f}", f"[{sev_color}]{a['severity']}[/{sev_color}]"
        )

    # Incidents
    incidents = slo_engine.incident_log[-5:][::-1]
    inc_table = Table(title="Recent Incidents", show_header=True, header_style="bold cyan",
                      show_edge=True, padding=(0, 1))
    inc_table.add_column("Event", min_width=20)
    inc_table.add_column("Job", min_width=10)
    inc_table.add_column("Failure Type", min_width=26)

    for inc in incidents:
        color = "green" if inc["event_type"] == "FAILURE_RESOLVED" else "red"
        inc_table.add_row(
            f"[{color}]{inc['event_type']}[/{color}]",
            inc["job_id"],
            inc["failure_type"],
        )

    console.clear()
    console.rule("[bold]⬡ GPUGuard — AI Training Infrastructure[/bold]")
    console.print(metrics_table)
    console.print()
    console.print(slo_table)
    console.print()
    console.print(jobs_table)
    console.print()
    console.print(anomaly_table)
    console.print()
    console.print(inc_table)


def render_plain(cluster, slo_engine, tick: int):
    slo_report = slo_engine.report()
    stats = cluster.get_cluster_stats()
    print(f"\n{'─'*60}")
    print(f"GPUGuard | tick={tick} | overall={slo_report.get('overall_status', 'OK')}")
    print(f"  Jobs: {stats['running_jobs']} running | GPU util: {stats['avg_gpu_utilization']:.1f}%")
    for name, slo in slo_report.get("slos", {}).items():
        print(f"  {name}: budget={slo['error_budget_remaining_pct']:.1f}% burn={slo['burn_rate']:.2f}x [{slo['status']}]")
    for job in cluster.jobs.values():
        print(f"  job={job.job_id} model={job.model_name} step={job.step}/{job.total_steps} loss={job.loss:.4f} [{job.status.value}]")


def main():
    parser = argparse.ArgumentParser(description="GPUGuard CLI — GPU Training Infrastructure Simulator")
    parser.add_argument("--ticks", type=int, default=0, help="Run N ticks then exit (0 = run forever)")
    parser.add_argument("--inject", choices=[f.value for f in FailureType if f != FailureType.NONE],
                        help="Inject a specific failure type on the first job")
    parser.add_argument("--export-metrics", action="store_true", help="Export Prometheus metrics text and exit")
    parser.add_argument("--interval", type=float, default=1.0, help="Tick interval in seconds (default: 1.0)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = parser.parse_args()

    cluster = GPUCluster(max_concurrent_jobs=5)
    slo = SLOEngine()
    remediation = AutoRemediationEngine()
    anomaly = AnomalyDetectionEngine()

    for _ in range(3):
        cluster.spawn_job()

    if args.inject:
        ft = FailureType(args.inject)
        first_job = next(iter(cluster.jobs.values()))
        first_job.failure_type = ft
        from simulator.gpu_job_simulator import JobStatus
        first_job.status = JobStatus.FAILED
        print(f"Injected failure: {args.inject} on job {first_job.job_id}")

    tick = 0
    try:
        while True:
            cluster.tick()
            slo_report = slo.evaluate(cluster)
            remediation.evaluate(cluster, slo_report)
            anomaly.evaluate(cluster)
            tick += 1

            if args.json:
                print(json.dumps({
                    "tick": tick,
                    "cluster": cluster.get_cluster_stats(),
                    "slo": slo_report,
                    "anomalies": anomaly.stats(),
                }, indent=2))
            elif HAS_RICH:
                render_rich(cluster, slo, remediation, anomaly, tick)
            else:
                render_plain(cluster, slo, tick)

            if args.ticks and tick >= args.ticks:
                print(f"\nCompleted {tick} ticks.")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nGPUGuard stopped.")
        print(f"Total ticks: {tick} | Failures: {cluster.total_failures} | Restarts: {cluster.total_restarts}")


if __name__ == "__main__":
    main()
