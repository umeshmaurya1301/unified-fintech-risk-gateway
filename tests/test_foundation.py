"""
tests/test_foundation.py — Pytest suite for Phase 2 + 3 (Models & Reset)
=========================================================================
Replaces verify_foundation.py with proper pytest test functions.
Satisfies M4: pytest-compatible test discovery (openenv validate, CI systems).
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# UFRGAction model validation
# ---------------------------------------------------------------------------

def test_ufrgaction_valid_construction():
    a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
    assert a.risk_decision == 1
    assert a.infra_routing == 2
    assert a.crypto_verify == 0


@pytest.mark.parametrize("bad_kwargs", [
    {"risk_decision": 3, "infra_routing": 0, "crypto_verify": 0},   # risk > 2
    {"risk_decision": 0, "infra_routing": 0, "crypto_verify": 2},   # crypto > 1
    {"risk_decision": -1, "infra_routing": 0, "crypto_verify": 0},  # negative
])
def test_ufrgaction_rejects_invalid(bad_kwargs):
    with pytest.raises(Exception):
        UFRGAction(**bad_kwargs)


# ---------------------------------------------------------------------------
# reset() — Gymnasium-standard signature
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return UnifiedFintechEnv()


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_returns_obs_and_info_tuple(env, task):
    result = env.reset(options={"task": task})
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_obs_type(env, task):
    obs, info = env.reset(options={"task": task})
    assert isinstance(obs, UFRGObservation)


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_info_task_key(env, task):
    _, info = env.reset(options={"task": task})
    assert info.get("task") == task


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_step_counter(env, task):
    env.reset(options={"task": task})
    assert env.current_step == 0


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_stores_current_task(env, task):
    env.reset(options={"task": task})
    assert env.current_task == task


def test_reset_default_task(env):
    """options=None should default to 'easy'."""
    _, info = env.reset()
    assert env.current_task == "easy"
    assert info["task"] == "easy"


def test_reset_seed_reproducible(env):
    obs1, _ = env.reset(seed=42, options={"task": "easy"})
    obs2, _ = env.reset(seed=42, options={"task": "easy"})
    assert obs1.risk_score == pytest.approx(obs2.risk_score, abs=1e-4)


def test_reset_invalid_task_raises(env):
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset(options={"task": "nightmare"})


# ---------------------------------------------------------------------------
# state() — non-destructive observation peek
# ---------------------------------------------------------------------------

def test_state_returns_ufrgobservation(env):
    env.reset(options={"task": "easy"})
    assert isinstance(env.state(), UFRGObservation)


def test_state_matches_reset_obs(env):
    obs, _ = env.reset(options={"task": "easy"})
    st = env.state()
    assert st.channel    == pytest.approx(obs.channel)
    assert st.risk_score == pytest.approx(obs.risk_score)
    assert st.kafka_lag  == pytest.approx(obs.kafka_lag)


# ---------------------------------------------------------------------------
# _generate_transaction — per-task distribution checks
# ---------------------------------------------------------------------------

def test_easy_risk_in_range(env):
    env.reset(seed=0, options={"task": "easy"})
    risks = [env._generate_transaction("easy").risk_score for _ in range(50)]
    assert all(5.0 <= r <= 30.0 for r in risks), f"Out-of-range: {[r for r in risks if not 5<=r<=30]}"


def test_hard_risk_always_high(env):
    env.reset(seed=0, options={"task": "hard"})
    risks = [env._generate_transaction("hard").risk_score for _ in range(50)]
    assert all(r >= 85.0 for r in risks), f"Low risk in hard: {min(risks):.1f}"


def test_medium_traffic_mix(env):
    env.reset(seed=0, options={"task": "medium"})
    events = []
    for _ in range(200):
        env._generate_transaction("medium")
        events.append(env._last_event_type)
    normal_pct = events.count("normal") / len(events) * 100
    flash_pct  = events.count("flash_sale") / len(events) * 100
    assert 55 < normal_pct < 95, f"Normal %: {normal_pct:.0f}"
    assert 5  < flash_pct  < 45, f"Flash %: {flash_pct:.0f}"
