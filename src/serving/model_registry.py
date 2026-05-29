"""
Model Registry & Inference Service
====================================
Centralized model versioning, storage, and serving infrastructure.

Provides:
  - Model registration with semantic versioning
  - Staged deployment (development → staging → production)
  - REST-like inference endpoint
  - A/B testing support
  - Model lineage tracking

Inspired by MLflow Model Registry patterns.
"""

import json
import time
import uuid
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Single registered model version."""
    model_name: str
    version: str
    stage: str = ModelStage.DEVELOPMENT.value
    artifact_path: str = ""
    run_id: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ModelRegistry:
    """
    Centralized model registry with version management.

    Stores model metadata and artifact paths. Enables:
      - Tracking which model version is in production
      - Rolling back to previous versions
      - Comparing metrics across versions
      - A/B testing between model versions

    Args:
        registry_dir: Directory for registry metadata
    """

    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "index.json"
        self._index: Dict[str, List[Dict]] = self._load_index()
        logger.info(f"ModelRegistry: {self.registry_dir}")

    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register(
        self,
        model_name: str,
        artifact_path: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Model family name (e.g., "node2vec_rl")
            artifact_path: Path to model artifact file
            metrics: Training/evaluation metrics
            tags: Metadata tags
            description: Human-readable description

        Returns:
            version: Registered ModelVersion
        """
        if model_name not in self._index:
            self._index[model_name] = []

        version_num = f"v{len(self._index[model_name]) + 1}.0.0"
        mv = ModelVersion(
            model_name=model_name,
            version=version_num,
            artifact_path=artifact_path,
            metrics=metrics or {},
            tags=tags or {},
            description=description,
        )
        self._index[model_name].append(asdict(mv))
        self._save_index()
        logger.info(f"Registered {model_name} {version_num} → {artifact_path}")
        return mv

    def transition_stage(
        self, model_name: str, version: str, new_stage: ModelStage
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Model family name
            version: Version string (e.g., "v1.0.0")
            new_stage: Target stage
        """
        versions = self._index.get(model_name, [])
        for mv in versions:
            if mv["version"] == version:
                mv["stage"] = new_stage.value
                mv["updated_at"] = time.time()
                self._save_index()
                logger.info(f"{model_name} {version} → {new_stage.value}")
                return
        raise ValueError(f"Model {model_name} {version} not found")

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Return the current production model version."""
        versions = self._index.get(model_name, [])
        prod = [v for v in versions if v["stage"] == ModelStage.PRODUCTION.value]
        return prod[-1] if prod else None

    def list_versions(self, model_name: str) -> List[Dict]:
        """List all registered versions for a model."""
        return self._index.get(model_name, [])

    def compare_versions(
        self, model_name: str, metric: str
    ) -> List[Dict]:
        """Return versions ranked by a metric."""
        versions = self.list_versions(model_name)
        eligible = [v for v in versions if metric in v.get("metrics", {})]
        return sorted(
            eligible,
            key=lambda v: v["metrics"][metric],
            reverse=True,
        )


class InferenceService:
    """
    HTTP inference server for deployed RL policies.

    Serves action predictions via a simple REST API:
      POST /predict  → {"state": <int>} → {"action": <int>, "q_values": [...]}
      GET  /health   → {"status": "ok"}
      GET  /model    → current model metadata

    Args:
        policy: {state: action} policy dict
        q_table: Optional Q-value tensor for uncertainty estimates
        port: HTTP server port
    """

    def __init__(
        self,
        policy: Dict[int, int],
        q_table: Optional[torch.Tensor] = None,
        port: int = 8080,
    ):
        self.policy = policy
        self.q_table = q_table
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._request_count = 0
        self._start_time = time.time()
        logger.info(f"InferenceService: port={port}")

    def predict(self, state: int) -> Dict[str, Any]:
        """
        Get policy action and optional Q-values for a state.

        Args:
            state: Node index

        Returns:
            prediction: {action, q_values (if available), latency_ms}
        """
        t0 = time.time()
        action = self.policy.get(state, 0)
        result = {
            "state": state,
            "action": action,
            "latency_ms": (time.time() - t0) * 1000,
        }
        if self.q_table is not None and state < len(self.q_table):
            result["q_values"] = self.q_table[state].tolist()
            result["confidence"] = float(
                torch.softmax(self.q_table[state], dim=0).max()
            )
        self._request_count += 1
        return result

    def start(self) -> None:
        """Start the HTTP inference server."""
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/predict":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    state = body.get("state", 0)
                    result = service.predict(state)
                    self._respond(200, result)

            def do_GET(self):
                if self.path == "/health":
                    self._respond(200, {
                        "status": "ok",
                        "uptime_s": time.time() - service._start_time,
                        "requests": service._request_count,
                    })

            def _respond(self, code, data):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        logger.info(f"InferenceService started on :{self.port}")

    def stop(self) -> None:
        """Shutdown the inference server."""
        if self._server:
            self._server.shutdown()
            logger.info("InferenceService stopped")
