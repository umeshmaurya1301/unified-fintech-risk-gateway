"""
tests/test_step.py — Pytest suite for Phase 4 (step() dynamics)
================================================================
Replaces verify_step.py with proper pytest test functions.
Satisfies M4: pytest-compatible test discovery (openenv validate, CI systems).
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv


def make_action(risk=0, infra=0, crypto=0) -> UFRGAction:
    return UFRGAction(risk_decision=risk, infra_routing=infra, crypto_verify=crypto)


@pytest.fixture
def env():
    return UnifiedFintechEnv()


@pytest.fixture
def easy_env(env):
    env.reset(options={"task": "easy"})
    return env


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

def test_step_returns_4_tuple(easy_env):
    result = easy_env.step(make_action())
    assert len(result) == 4


def test_step_obs_type(easy_env):
    obs, _, _, _ = easy_env.step(make_action())
    assert isinstance(obs, UFRGObservation)


def test_step_reward_float(easy_env):
    _, reward, _, _ = easy_env.step(make_action())
    assert isinstance(reward, float)


def test_step_done_bool(easy_env):
    _, _, done, _ = easy_env.step(make_action())
    assert isinstance(done, bool)


def test_step_info_dict(easy_env):
    _, _, _, info = easy_env.step(make_action())
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Reward clamping [0.0, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reward_always_in_range(env, task):
    env.reset(options={"task": task})
    _, reward, _, _ = env.step(make_action())
    assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0, 1]"


# ---------------------------------------------------------------------------
# Baseline reward (~0.8, no penalties)
# ---------------------------------------------------------------------------

def test_baseline_reward(easy_env):
    easy_env._rolling_latency = 10.0
    easy_env._rolling_lag = 0.0
    # Force a clean low-p99 obs
    easy_env._current_obs = UFRGObservation(
        channel=easy_env._current_obs.channel,
        risk_score=easy_env._current_obs.risk_score,
        kafka_lag=0.0,
        api_latency=10.0,
        rolling_p99=10.0,
    )
    _, r, _, info = easy_env.step(make_action(risk=0, infra=0, crypto=0))
    assert 0.5 <= r <= 0.8, f"Baseline reward {r}, raw={info['reward_raw']}"


# ---------------------------------------------------------------------------
# Throttle penalty (-0.2)
# ---------------------------------------------------------------------------

def test_throttle_penalty(env):
    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    env._current_obs = UFRGObservation(
        channel=0.0, risk_score=10.0, kafka_lag=0.0,
        api_latency=10.0, rolling_p99=10.0,
    )
    _, r_normal, _, _ = env.step(make_action(infra=0))

    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    env._current_obs = UFRGObservation(
        channel=0.0, risk_score=10.0, kafka_lag=0.0,
        api_latency=10.0, rolling_p99=10.0,
    )
    _, r_throttle, _, _ = env.step(make_action(infra=1))

    assert abs((r_normal - r_throttle) - 0.2) < 0.05, \
        f"normal={r_normal:.3f}, throttle={r_throttle:.3f}"


# ---------------------------------------------------------------------------
# Circuit-breaker penalty (-0.5) and accumulator reset
# ---------------------------------------------------------------------------

def test_circuit_breaker_penalty(easy_env):
    easy_env._rolling_latency = 10.0
    easy_env._rolling_lag = 0.0
    _, _, _, info = easy_env.step(make_action(infra=2))
    assert abs(info["reward_raw"] - 0.3) < 0.05, f"raw={info['reward_raw']:.3f}"
    assert info["circuit_breaker_tripped"]
    assert easy_env._rolling_lag == 0.0


def test_circuit_breaker_resets_accumulators(easy_env):
    easy_env._rolling_lag = 3000.0
    easy_env.step(make_action(infra=2))
    assert easy_env._rolling_lag == 0.0
    assert abs(easy_env._rolling_latency - 50.0) < 30.0


# ---------------------------------------------------------------------------
# SLA breach penalty (-0.3)
# ---------------------------------------------------------------------------

def test_sla_breach_penalty(easy_env):
    easy_env._rolling_lag = 0.0
    easy_env._current_obs = UFRGObservation(
        channel=easy_env._current_obs.channel,
        risk_score=easy_env._current_obs.risk_score,
        kafka_lag=0.0,
        api_latency=easy_env._current_obs.api_latency,
        rolling_p99=2000.0,   # well above 800ms SLA threshold
    )
    _, _, _, info = easy_env.step(make_action(infra=0))
    assert abs(info["reward_raw"] - 0.5) < 0.1, f"raw={info['reward_raw']:.3f}"


# ---------------------------------------------------------------------------
# Catastrophic fraud gate (-1.0)
# ---------------------------------------------------------------------------

def test_fraud_gate_zeroes_reward(env):
    env.reset(options={"task": "hard"})   # hard always has risk > 85
    _, r_fraud, _, info = env.step(make_action(risk=0, infra=0, crypto=1))  # Approve+SkipVerify
    assert r_fraud == 0.0, f"reward={r_fraud}, raw={info['reward_raw']:.3f}"
    assert info["obs_risk_score"] > 80.0


# ---------------------------------------------------------------------------
# Crash condition
# ---------------------------------------------------------------------------

def test_crash_zeroes_reward_and_sets_done(easy_env):
    easy_env._rolling_lag = 4500.0
    easy_env._rolling_latency = 10.0
    _, r, done, _ = easy_env.step(make_action(infra=0))
    assert r == 0.0
    assert done is True


def test_circuit_breaker_prevents_crash(easy_env):
    easy_env._rolling_lag = 4500.0
    easy_env._rolling_latency = 10.0
    _, _, done, info = easy_env.step(make_action(infra=2))
    # CB resets lag to 0 — crash cannot fire
    assert info["internal_rolling_lag"] < 50.0


# ---------------------------------------------------------------------------
# max_steps triggers done
# ---------------------------------------------------------------------------

def test_max_steps_triggers_done(easy_env):
    easy_env.max_steps = 3
    for _ in range(2):
        _, _, done, _ = easy_env.step(make_action())
        assert not done
    _, _, done, _ = easy_env.step(make_action())
    assert done


# ---------------------------------------------------------------------------
# Info dict completeness
# ---------------------------------------------------------------------------

REQUIRED_INFO_KEYS = {
    "step", "task", "event_type",
    "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
    "action_risk_decision", "action_infra_routing", "action_crypto_verify",
    "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
    "internal_rolling_lag", "internal_rolling_latency",
}

def test_info_dict_has_required_keys(env):
    env.reset(options={"task": "medium"})
    _, _, _, info = env.step(make_action())
    missing = REQUIRED_INFO_KEYS - info.keys()
    assert not missing, f"Missing info keys: {missing}"
