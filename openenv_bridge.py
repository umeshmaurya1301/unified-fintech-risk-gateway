"""
openenv_bridge.py — Python Bridge for OpenEnv CLI Compatibility
================================================================
The Meta OpenEnv Hackathon's ``openenv validate`` CLI command requires a
Python-importable entry point defined in ``openenv.yaml``:

    entry_point: "unified_gateway:UnifiedFintechEnv"

This bridge satisfies that contract by providing a thin ``UnifiedFintechEnv``
class that forwards all Gymnasium calls to the running Spring Boot REST API.

This means the Java server is the *actual* environment — Python is only a
transport adapter to satisfy the CLI tool.

Environment variables
---------------------
  SPACE_URL   Base URL of the Spring Boot server (default: http://localhost:7860)

Usage (inside Docker or locally with the server already running)
---------------------------------------------------------------
  python openenv_bridge.py          # quick self-test
  openenv validate .                # full validator (uses this as entry_point)
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────
_SPACE_URL: str = os.environ.get("SPACE_URL", "http://localhost:7860").rstrip("/")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    url  = f"{_SPACE_URL}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    url = f"{_SPACE_URL}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# ── Minimal Pydantic-style models for openenv validate compatibility ──────────

class UFRGObservation:
    """Thin observation wrapper returned by reset() and step()."""
    __slots__ = ("channel", "risk_score", "kafka_lag", "api_latency", "rolling_p99")

    def __init__(self, channel, risk_score, kafka_lag, api_latency, rolling_p99):
        self.channel     = float(channel)
        self.risk_score  = float(risk_score)
        self.kafka_lag   = float(kafka_lag)
        self.api_latency = float(api_latency)
        self.rolling_p99 = float(rolling_p99)

    @classmethod
    def from_dict(cls, d: dict) -> "UFRGObservation":
        return cls(
            channel     = d.get("channel",     0.0),
            risk_score  = d.get("risk_score",  0.0),
            kafka_lag   = d.get("kafka_lag",   0.0),
            api_latency = d.get("api_latency", 0.0),
            rolling_p99 = d.get("rolling_p99", 0.0),
        )

    def __repr__(self):
        return (f"UFRGObservation(channel={self.channel}, risk_score={self.risk_score}, "
                f"kafka_lag={self.kafka_lag}, api_latency={self.api_latency}, "
                f"rolling_p99={self.rolling_p99})")


class UFRGAction:
    """Thin action wrapper; validates ranges matching the Java @Min/@Max constraints."""
    VALID = {
        "risk_decision": (0, 2),
        "infra_routing": (0, 2),
        "crypto_verify": (0, 1),
    }
    __slots__ = ("risk_decision", "infra_routing", "crypto_verify")

    def __init__(self, risk_decision: int, infra_routing: int, crypto_verify: int):
        for attr, (lo, hi) in self.VALID.items():
            val = locals()[attr]
            if not (lo <= val <= hi):
                raise ValueError(f"{attr}={val} out of range [{lo}, {hi}]")
        self.risk_decision = risk_decision
        self.infra_routing = infra_routing
        self.crypto_verify = crypto_verify

    def to_dict(self) -> dict:
        return {
            "risk_decision": self.risk_decision,
            "infra_routing": self.infra_routing,
            "crypto_verify": self.crypto_verify,
        }


# ── Main bridge environment ───────────────────────────────────────────────────

class UnifiedFintechEnv:
    """
    OpenEnv-compatible Gymnasium environment that proxies all calls to the
    Java Spring Boot REST API.

    This is the ``entry_point`` referenced in ``openenv.yaml``.  The openenv
    validator calls ``reset()``, ``step()``, and ``state()``; all responses
    are forwarded transparently to the running Java server.
    """

    # openenv validator reads these attributes
    reward_range = (0.0, 1.0)

    def __init__(self):
        self._current_obs: UFRGObservation | None = None
        self.current_task: str  = "easy"
        self.current_step: int  = 0
        self.max_steps:    int  = 100

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        task = "easy"
        if options and isinstance(options.get("task"), str):
            task = options["task"]
        if task not in ("easy", "medium", "hard"):
            raise ValueError(f"Invalid task '{task}'. Must be easy, medium, or hard.")

        data = _post("/reset", {"task": task})
        self._current_obs  = UFRGObservation.from_dict(data["observation"])
        self.current_task  = task
        self.current_step  = 0
        info               = data.get("info", {"task": task})
        return self._current_obs, info

    def step(self, action: UFRGAction):
        data = _post("/step", {"action": action.to_dict()})
        self._current_obs = UFRGObservation.from_dict(data["observation"])
        reward: float     = float(data["reward"])
        done:   bool      = bool(data["done"])
        info:   dict      = data.get("info", {})
        self.current_step += 1
        return self._current_obs, reward, done, info

    def state(self) -> UFRGObservation:
        if self._current_obs is not None:
            return self._current_obs
        data = _get("/state")
        self._current_obs = UFRGObservation.from_dict(data["observation"])
        return self._current_obs

    def close(self):
        pass   # no persistent connection to close


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"[bridge] Testing connection to {_SPACE_URL} ...")
    try:
        env = UnifiedFintechEnv()
        obs, info = env.reset(options={"task": "easy"})
        print(f"[bridge] reset() OK  → {obs}")
        action = UFRGAction(risk_decision=0, infra_routing=0, crypto_verify=1)
        obs2, reward, done, info2 = env.step(action)
        print(f"[bridge] step()  OK  → reward={reward:.3f} done={done}")
        obs3 = env.state()
        print(f"[bridge] state() OK  → {obs3}")
        print("[bridge] ✅ All checks passed — openenv_bridge is functional.")
    except Exception as exc:
        print(f"[bridge] ❌ FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
