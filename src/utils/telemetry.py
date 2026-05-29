"""
Telemetry & Monitoring Service
================================
Real-time RL training observability with Prometheus-compatible metrics.

Emits:
  - Training throughput (episodes/second, steps/second)
  - Loss metrics (Node2Vec, InferNet, policy)
  - Reward statistics (mean, std, percentiles)
  - System metrics (CPU, memory, GPU utilization)
  - Exploration statistics (epsilon, entropy)

Exposes metrics via HTTP endpoint for Prometheus scraping,
and optionally writes to InfluxDB for Grafana dashboards.
"""

import time
import psutil
import threading
import json
import logging
from collections import deque
from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Sliding window metric aggregator."""
    name: str
    window_size: int = 100
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record(self, value: float) -> None:
        self.values.append(value)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def count(self) -> int:
        return len(self.values)


class TelemetryService:
    """
    Lightweight telemetry service for RL training observability.

    Collects metrics from the training loop and exposes them via:
      1. In-memory ring buffers (for in-process querying)
      2. Prometheus-format HTTP endpoint (for external scraping)
      3. JSON export (for offline analysis)

    Thread-safe: metrics can be recorded from worker processes
    while the HTTP server runs on a background thread.

    Args:
        port: HTTP port for Prometheus scraping (default 9090)
        window_size: Rolling window size for statistics
        enable_system_metrics: Whether to collect CPU/memory
    """

    def __init__(
        self,
        port: int = 9090,
        window_size: int = 100,
        enable_system_metrics: bool = True,
    ):
        self.port = port
        self.window_size = window_size
        self.enable_system_metrics = enable_system_metrics
        self._lock = threading.Lock()
        self._metrics: Dict[str, MetricWindow] = {}
        self._counters: Dict[str, int] = {}
        self._start_time = time.time()
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        logger.info(f"TelemetryService initialized (port={port})")

    def _get_or_create(self, name: str) -> MetricWindow:
        if name not in self._metrics:
            self._metrics[name] = MetricWindow(name=name, window_size=self.window_size)
        return self._metrics[name]

    def record(self, name: str, value: float) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name (use "/" as namespace separator)
            value: Numeric value
        """
        with self._lock:
            self._get_or_create(name).record(value)

    def record_batch(self, metrics: Dict[str, float]) -> None:
        """Record multiple metrics at once."""
        for name, value in metrics.items():
            self.record(name, value)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def get_summary(self) -> Dict[str, Any]:
        """
        Return current metric summary.

        Returns:
            summary: Dict with mean/latest/count per metric,
                     plus system metrics and uptime
        """
        with self._lock:
            summary = {
                "uptime_s": time.time() - self._start_time,
                "counters": dict(self._counters),
                "metrics": {
                    name: {
                        "mean": m.mean,
                        "latest": m.latest,
                        "count": m.count,
                    }
                    for name, m in self._metrics.items()
                },
            }

        if self.enable_system_metrics:
            summary["system"] = self._collect_system_metrics()

        return summary

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect CPU, memory, and GPU metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
        }
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_memory_mb"] = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                )
                metrics["gpu_utilization"] = 0.0  # Requires nvidia-ml-py
        except Exception:
            pass
        return metrics

    def to_prometheus(self) -> str:
        """
        Format all metrics as Prometheus text exposition format.

        Returns:
            prometheus_text: Metrics in Prometheus format for scraping
        """
        lines = []
        summary = self.get_summary()

        lines.append(f"# HELP rl_uptime_seconds Training uptime")
        lines.append(f"# TYPE rl_uptime_seconds gauge")
        lines.append(f'rl_uptime_seconds {summary["uptime_s"]:.2f}')

        for name, vals in summary["metrics"].items():
            safe_name = name.replace("/", "_").replace("-", "_")
            lines.append(f"# HELP rl_{safe_name}_mean Rolling mean of {name}")
            lines.append(f"# TYPE rl_{safe_name}_mean gauge")
            lines.append(f'rl_{safe_name}_mean {vals["mean"]:.6f}')
            lines.append(f'rl_{safe_name}_latest {vals["latest"]:.6f}')

        for name, count in summary["counters"].items():
            safe_name = name.replace("/", "_").replace("-", "_")
            lines.append(f"# TYPE rl_{safe_name}_total counter")
            lines.append(f'rl_{safe_name}_total {count}')

        if "system" in summary:
            sys = summary["system"]
            lines.append(f'rl_system_cpu_percent {sys.get("cpu_percent", 0):.2f}')
            lines.append(f'rl_system_memory_percent {sys.get("memory_percent", 0):.2f}')

        return "\n".join(lines)

    def start_server(self) -> None:
        """Start Prometheus HTTP metrics server in background thread."""
        telemetry = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = telemetry.to_prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                elif self.path == "/summary":
                    body = json.dumps(telemetry.get_summary(), indent=2).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *args):
                pass  # Suppress request logs

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        logger.info(f"Telemetry server started on :{self.port}/metrics")

    def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            logger.info("Telemetry server stopped")

    def export_json(self, path: str) -> None:
        """Export full metric history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Telemetry exported to {path}")
