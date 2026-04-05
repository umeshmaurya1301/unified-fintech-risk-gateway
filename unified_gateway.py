"""
Unified Fintech Risk Gateway — Gymnasium Environment
=====================================================
Observation space  : Box(5,) float32
  [0] Channel          — encoded channel ID      [0,    2]
  [1] Risk Score       — normalized risk signal  [0.0, 100.0]
  [2] Kafka Lag        — consumer lag in msgs    [0,    10000]
  [3] API Latency      — request latency (ms)    [0.0,  5000.0]
  [4] Rolling P99 SLA  — P99 latency (ms)        [0.0,  5000.0]

Action space       : MultiDiscrete([3, 3, 2])
  [0] Risk Decision    — 0=APPROVE  1=REVIEW    2=REJECT
  [1] Infra Routing    — 0=PRIMARY  1=SECONDARY 2=FALLBACK
  [2] Crypto Verify    — 0=SKIP     1=VERIFY
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UnifiedFintechRiskGateway(gym.Env):
    """
    A Gymnasium environment that models a unified fintech risk gateway.

    The agent observes five real-time signals across a payment channel and
    must simultaneously decide:
      - the risk disposition of an incoming transaction,
      - how to route the request across available infrastructure tiers, and
      - whether cryptographic verification is required.

    Episode length is capped at ``self.max_steps`` steps.
    """

    # ------------------------------------------------------------------
    # Metadata (no render modes implemented yet)
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Episode configuration
        # ------------------------------------------------------------------
        self.max_steps: int = 1000

        # ------------------------------------------------------------------
        # Observation space — Box(5,) dtype=float32
        #
        # Columns (in order):
        #   0  Channel          integer-coded channel id    [0,      2     ]
        #   1  Risk Score       raw risk signal             [0.0,  100.0   ]
        #   2  Kafka Lag        consumer-group message lag  [0,    10 000  ]
        #   3  API Latency      end-to-end latency (ms)     [0.0,  5 000.0 ]
        #   4  Rolling P99 SLA  P99 latency bucket (ms)     [0.0,  5 000.0 ]
        # ------------------------------------------------------------------
        obs_low = np.array(
            [0.0,    0.0,     0.0,    0.0,    0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [2.0,  100.0, 10000.0, 5000.0, 5000.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(5,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space — MultiDiscrete([3, 3, 2])
        #
        # Dimensions (in order):
        #   0  Risk Decision   {0: APPROVE, 1: REVIEW, 2: REJECT}
        #   1  Infra Routing   {0: PRIMARY, 1: SECONDARY, 2: FALLBACK}
        #   2  Crypto Verify   {0: SKIP,    1: VERIFY}
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([3, 3, 2], dtype=np.int64),
        )

        # Internal step counter — incremented in step(), reset in reset()
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # reset — Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state and return the first
        observation together with an empty info dict.

        Calling ``super().reset(seed=seed)`` seeds ``self.np_random`` so
        that all subsequent calls to ``_generate_transaction`` are
        reproducible when a seed is supplied.

        Returns
        -------
        observation : np.ndarray, shape (5,), dtype float32
            The initial observation produced by the Synthetic Data Engine.
        info : dict
            Empty auxiliary info dict (Gymnasium API requirement).
        """
        # Seed self.np_random via the Gymnasium base class.
        super().reset(seed=seed)

        # ---- Episode counters ----------------------------------------
        self._current_step: int = 0

        # ---- Rolling-window accumulators (used by step() later) -------
        # These track an exponential moving average of Kafka Lag and API
        # Latency across steps, giving the agent a smoothed P99 signal.
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = 0.0

        # ---- Generate the first transaction observation ----------------
        self.state: np.ndarray = self._generate_transaction()

        return self.state, {}

    # ------------------------------------------------------------------
    # _generate_transaction — Synthetic Data Engine
    # ------------------------------------------------------------------
    def _generate_transaction(self) -> np.ndarray:
        """
        Produce a single synthetic transaction observation vector.

        Event distribution
        ------------------
        - 80 % → **Normal Traffic**   low risk, stable infra
        - 10 % → **Flash Sale**        very low risk, heavy Kafka & latency spike
        - 10 % → **Botnet Attack**     extreme risk, moderate infra stress

        All outputs are clipped to the declared ``observation_space`` bounds
        and cast to ``np.float32``.

        Returns
        -------
        obs : np.ndarray, shape (5,), dtype float32
            [channel, risk_score, kafka_lag, api_latency, rolling_p99_sla]
        """
        rng = self.np_random  # gymnasium-seeded Generator

        # ---- Event trigger (draw a single uniform float) ---------------
        roll: float = rng.uniform(0.0, 1.0)

        if roll < 0.80:
            # ==============================================================
            # NORMAL TRAFFIC — 80 %
            # Low risk, stable and low infrastructure load.
            # ==============================================================
            channel      = float(rng.integers(0, 3))          # {0, 1, 2}
            risk_score   = rng.uniform(5.0, 30.0)
            kafka_lag    = rng.uniform(0.0, 500.0)
            api_latency  = rng.uniform(50.0, 300.0)
            p99_sla      = rng.uniform(80.0, 400.0)
            event_type   = "normal"

        elif roll < 0.90:
            # ==============================================================
            # FLASH SALE — 10 %
            # Massive transaction volume; risk is very low (legitimate users)
            # but Kafka consumer lag spikes heavily and API latency degrades.
            # ==============================================================
            channel      = float(rng.integers(0, 3))
            risk_score   = rng.uniform(0.0, 10.0)
            kafka_lag    = rng.uniform(3000.0, 8000.0)
            api_latency  = rng.uniform(800.0, 3000.0)
            p99_sla      = rng.uniform(1200.0, 4000.0)
            event_type   = "flash_sale"

        else:
            # ==============================================================
            # BOTNET ATTACK — 10 %
            # Extreme risk signal; infra is stressed but deliberately does
            # NOT max out (attackers throttle to avoid detection).
            # ==============================================================
            channel      = float(rng.integers(0, 3))
            risk_score   = rng.uniform(85.0, 100.0)
            kafka_lag    = rng.uniform(500.0, 3000.0)
            api_latency  = rng.uniform(300.0, 1500.0)
            p99_sla      = rng.uniform(500.0, 2000.0)
            event_type   = "botnet_attack"

        # ---- Update rolling accumulators (EMA, α = 0.2) ----------------
        # Smooths out per-step noise so the agent sees a stable P99 trend.
        alpha: float = 0.2
        self._rolling_lag     = alpha * kafka_lag   + (1.0 - alpha) * self._rolling_lag
        self._rolling_latency = alpha * api_latency + (1.0 - alpha) * self._rolling_latency

        # Use the EMA latency as the Rolling P99 SLA observation so that
        # the agent gets a smoothed signal rather than a raw point sample.
        # Clamp to the declared SLA high bound just in case.
        smoothed_p99 = min(self._rolling_latency, 5000.0)

        # ---- Build & clip the observation vector -----------------------
        obs = np.array(
            [channel, risk_score, kafka_lag, api_latency, smoothed_p99],
            dtype=np.float32,
        )
        obs = np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

        # Attach event metadata to the instance so step() can read it
        # for reward shaping without re-exposing it in the obs vector.
        self._last_event_type: str = event_type

        return obs

    # ------------------------------------------------------------------
    # step — Gymnasium API
    # ------------------------------------------------------------------
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Run one time-step of the environment's dynamics.

        Parameters
        ----------
        action : np.ndarray, shape (3,), dtype int64
            A sampled or agent-selected action from ``action_space``.
            action[0] — Risk Decision  (0=APPROVE, 1=REVIEW, 2=REJECT)
            action[1] — Infra Routing  (0=PRIMARY, 1=SECONDARY, 2=FALLBACK)
            action[2] — Crypto Verify  (0=SKIP,    1=VERIFY)

        Returns
        -------
        observation  : np.ndarray, shape (5,), dtype float32
        reward       : float
        terminated   : bool   — True when a terminal state is reached
        truncated    : bool   — True when ``max_steps`` is exceeded
        info         : dict   — Auxiliary diagnostic information
        """
        # TODO: implement transition dynamics, reward shaping, and termination logic
        raise NotImplementedError
