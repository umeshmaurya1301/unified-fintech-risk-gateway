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
        Reset the environment to an initial state.

        Returns
        -------
        observation : np.ndarray, shape (5,), dtype float32
            The initial observation sampled from ``observation_space``.
        info : dict
            Auxiliary diagnostic information (empty for now).
        """
        super().reset(seed=seed)
        # TODO: implement initial state logic
        raise NotImplementedError

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
