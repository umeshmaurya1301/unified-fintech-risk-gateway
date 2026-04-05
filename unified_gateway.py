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
  [0] Risk Decision    — 0=APPROVE  1=REJECT           2=CHALLENGE (PIN reprompt)
  [1] Infra Routing    — 0=ROUTE_NORMAL  1=THROTTLE    2=CIRCUIT_BREAKER
  [2] Crypto Verify    — 0=FULL_VERIFY (slow)  1=SKIP_VERIFY (fast)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UnifiedFintechEnv(gym.Env):
    """
    Gymnasium environment modelling a unified fintech risk gateway.

    The agent observes five real-time signals across a payment channel and
    must simultaneously decide:
      - the risk disposition of an incoming transaction,
      - how to route the request across available infrastructure tiers, and
      - whether cryptographic verification is required.

    Previously known as ``UnifiedFintechRiskGateway``.
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
        #   0  Risk Decision   {0: APPROVE, 1: REJECT, 2: CHALLENGE (PIN reprompt)}
        #   1  Infra Routing   {0: ROUTE_NORMAL, 1: THROTTLE, 2: CIRCUIT_BREAKER}
        #   2  Crypto Verify   {0: FULL_VERIFY (slow), 1: SKIP_VERIFY (fast)}
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
            action[0] — Risk Decision  (0=APPROVE, 1=REJECT, 2=CHALLENGE)
            action[1] — Infra Routing  (0=ROUTE_NORMAL, 1=THROTTLE, 2=CIRCUIT_BREAKER)
            action[2] — Crypto Verify  (0=FULL_VERIFY, 1=SKIP_VERIFY)

        Returns
        -------
        observation  : np.ndarray, shape (5,), dtype float32
            Next transaction observation from the Synthetic Data Engine.
        reward       : float
            Composite reward reflecting risk accuracy, crypto cost, infra
            health, and SLA compliance.
        terminated   : bool
            True on a system crash (kafka_lag > 4 000 without circuit breaker)
            or when ``max_steps`` is reached.
        truncated    : bool
            Always False — episode truncation is folded into ``terminated``.
        info         : dict
            Auxiliary diagnostics (event type, component rewards, flags).
        """
        # ------------------------------------------------------------------
        # Unpack action dimensions
        # ------------------------------------------------------------------
        risk_decision: int = int(action[0])  # 0=APPROVE  1=REJECT  2=CHALLENGE
        infra_routing: int = int(action[1])  # 0=ROUTE_NORMAL  1=THROTTLE  2=CIRCUIT_BREAKER
        crypto_verify: int = int(action[2])  # 0=FULL_VERIFY  1=SKIP_VERIFY

        # ------------------------------------------------------------------
        # Unpack current observation
        # ------------------------------------------------------------------
        _channel, risk_score, kafka_lag, _api_latency, rolling_p99 = (
            float(v) for v in self.state
        )

        # ------------------------------------------------------------------
        # Reward accumulators & flags
        # ------------------------------------------------------------------
        total_reward: float = 0.0
        terminated:   bool  = False
        circuit_breaker_tripped: bool = False

        # Partial rewards stored for the info dict
        reward_risk:   float = 0.0
        reward_crypto: float = 0.0
        reward_infra:  float = 0.0
        reward_sla:    float = 0.0
        reward_crash:  float = 0.0

        # ------------------------------------------------------------------
        # ① Risk Decision reward
        #    Penalties/bonuses depend on whether this is a high-risk txn.
        # ------------------------------------------------------------------
        if risk_score > 80.0:
            # High-risk transaction — the agent must not approve blindly.
            if risk_decision == 0:      # APPROVE  → catastrophic miss
                reward_risk = -150.0
            elif risk_decision == 1:    # REJECT   → correct decisive action
                reward_risk =  +30.0
            else:                       # CHALLENGE → cautious, earns partial credit
                reward_risk =  +15.0
        else:
            # Low-risk transaction — blocking legitimate traffic has a cost.
            if risk_decision == 0:      # APPROVE  → correct, frictionless
                reward_risk = +10.0
            elif risk_decision == 1:    # REJECT   → false positive, hurts UX
                reward_risk = -20.0
            else:                       # CHALLENGE → unnecessary friction
                reward_risk =  -5.0

        total_reward += reward_risk

        # ------------------------------------------------------------------
        # ② Crypto Verify cost / fraud gate
        #    FULL_VERIFY adds lag & latency pressure on the accumulator.
        #    SKIP_VERIFY is faster but must never be paired with APPROVE on
        #    a high-risk transaction — doing so triggers a fraud penalty.
        # ------------------------------------------------------------------
        if crypto_verify == 0:          # FULL_VERIFY — thorough but expensive
            self._rolling_lag     += 150.0
            self._rolling_latency += 200.0
        else:                           # SKIP_VERIFY — fast path
            self._rolling_lag -= 100.0
            # Critical fraud gate: SKIP + APPROVE on high-risk = catastrophic
            if risk_decision == 0 and risk_score > 80.0:
                reward_crypto = -200.0
                total_reward  += reward_crypto

        # ------------------------------------------------------------------
        # ③ Infrastructure Routing & Circuit Breaker
        #    Actions directly mutate the EMA accumulators so their effect
        #    propagates organically into the next _generate_transaction() call.
        # ------------------------------------------------------------------
        if infra_routing == 0:          # ROUTE_NORMAL — standard path, adds load
            self._rolling_lag += 100.0

        elif infra_routing == 1:        # THROTTLE — sheds load; penalises good traffic
            self._rolling_lag -= 300.0
            reward_infra  = -10.0
            total_reward += reward_infra

        else:                           # CIRCUIT_BREAKER — nuclear option
            reward_infra  = -100.0
            total_reward += reward_infra
            # Instantly heal internal state to a safe baseline
            self._rolling_lag     = 0.0
            self._rolling_latency = 50.0
            circuit_breaker_tripped = True

        # Guard: accumulators must never go negative
        self._rolling_lag     = max(0.0, self._rolling_lag)
        self._rolling_latency = max(0.0, self._rolling_latency)

        # ------------------------------------------------------------------
        # ④ Global SLA degradation penalty  (evaluated on current-tick obs)
        # ------------------------------------------------------------------
        if rolling_p99 > 800.0:
            reward_sla   = -20.0
            total_reward += reward_sla

        # ------------------------------------------------------------------
        # ④ Kafka crash penalty  (evaluated on current-tick obs)
        #    The circuit breaker is the one sanctioned escape hatch — if it
        #    was tripped this tick the lag has already been zeroed, so we
        #    skip the crash check to avoid double-penalising the agent.
        # ------------------------------------------------------------------
        if kafka_lag > 4000.0 and not circuit_breaker_tripped:
            reward_crash = -500.0
            total_reward += reward_crash
            terminated    = True

        # ------------------------------------------------------------------
        # Advance episode
        # ------------------------------------------------------------------
        self._current_step += 1

        # Produce the next observation; the mutated EMA accumulators carry
        # the agent's infra decisions forward into the next transaction.
        self.state = self._generate_transaction()

        # Max-steps termination (folded into terminated per spec)
        if self._current_step >= self.max_steps:
            terminated = True

        # ------------------------------------------------------------------
        # Info dict — structured diagnostics for logging / debugging
        # ------------------------------------------------------------------
        info: dict[str, Any] = {
            # Episode progress
            "step":                    self._current_step,
            "event_type":              self._last_event_type,
            # Observation snapshot that drove the decisions this tick
            "obs_risk_score":          risk_score,
            "obs_kafka_lag":           kafka_lag,
            "obs_rolling_p99":         rolling_p99,
            # Action taken
            "action_risk_decision":    risk_decision,
            "action_infra_routing":    infra_routing,
            "action_crypto_verify":    crypto_verify,
            # Reward breakdown
            "reward_risk":             reward_risk,
            "reward_crypto":           reward_crypto,
            "reward_infra":            reward_infra,
            "reward_sla":              reward_sla,
            "reward_crash":            reward_crash,
            "total_reward":            total_reward,
            # Flags
            "circuit_breaker_tripped": circuit_breaker_tripped,
            "terminated":              terminated,
            # Post-action accumulator state
            "internal_rolling_lag":    self._rolling_lag,
            "internal_rolling_latency":self._rolling_latency,
        }

        return self.state, float(total_reward), terminated, False, info
