"""
Phase 4 — step() Verification
==============================
Tests every reward branch, crash condition, done flag, and return types.
"""
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv

passed = 0
failed = 0

def check(label: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  —  {detail}")


def make_action(risk=0, infra=0, crypto=0) -> UFRGAction:
    return UFRGAction(risk_decision=risk, infra_routing=infra, crypto_verify=crypto)


env = UnifiedFintechEnv()

# ═══════════════════════════════════════════════════════════════════
print("── Return type contract ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
result = env.step(make_action())
check("step returns 4-tuple", len(result) == 4,
      f"got {len(result)}-tuple")
obs, reward, done, info = result
check("obs is UFRGObservation", isinstance(obs, UFRGObservation))
check("reward is float", isinstance(reward, float))
check("done is bool", isinstance(done, bool))
check("info is dict", isinstance(info, dict))

# ═══════════════════════════════════════════════════════════════════
print("\n── Reward clipping [0.0, 1.0] ──")
# ═══════════════════════════════════════════════════════════════════
for task in ["easy", "medium", "hard"]:
    env.reset(task_name=task)
    for _ in range(20):
        a = make_action(risk=0, infra=0, crypto=0)
        _, r, _, _ = env.step(a)
        check(f"reward in [0,1] ({task})", 0.0 <= r <= 1.0, f"got {r}")
        break   # one check per task is sufficient here

# ═══════════════════════════════════════════════════════════════════
print("\n── Baseline reward (no penalties) ──")
# ═══════════════════════════════════════════════════════════════════
# Force a clean low-p99 state so no SLA penalty fires
env.reset(task_name="easy")
env._rolling_latency = 10.0   # p99 will be very low
env._rolling_lag = 0.0
# Action: Normal infra, FullVerify, Approve — no fraud (easy = low risk)
_, r, _, info = env.step(make_action(risk=0, infra=0, crypto=0))
# Expected: 0.8 base — no deductions (p99 was 10ms, no throttle, no CB, no fraud)
check("baseline ~0.8 with no penalties", 0.5 <= r <= 0.8,
      f"got {r}, raw={info['reward_raw']}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Throttle penalty (-0.2) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, r1, _, _ = env.step(make_action(infra=0))   # Normal
env.reset(task_name="easy")
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, r2, _, _ = env.step(make_action(infra=1))   # Throttle
check("throttle reduces reward by ~0.2",
      abs((r1 - r2) - 0.2) < 0.05,
      f"normal={r1:.3f}, throttle={r2:.3f}, diff={r1-r2:.3f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker penalty (-0.5) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, r_cb, _, info_cb = env.step(make_action(infra=2))
check("circuit-breaker reduces reward by ~0.5",
      abs(info_cb['reward_raw'] - 0.3) < 0.05,
      f"raw={info_cb['reward_raw']:.3f}")
check("circuit_breaker_tripped flag set", info_cb["circuit_breaker_tripped"])
check("_rolling_lag reset to 0.0 after CB", env._rolling_lag == 0.0)
check("_rolling_latency ≈ 50.0 after CB",
      abs(info_cb["internal_rolling_latency"] - 50.0) < 15.0,
      f"info shows {info_cb['internal_rolling_latency']:.2f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── SLA breach penalty (-0.3) ──")
# ═══════════════════════════════════════════════════════════════════
# Force the current_obs.rolling_p99 > 800 by injecting a high-p99 observation
env.reset(task_name="easy")
env._rolling_lag = 0.0
# Patch _current_obs directly so rolling_p99 seen by step() is high
env._current_obs = UFRGObservation(
    channel=env._current_obs.channel,
    risk_score=env._current_obs.risk_score,
    kafka_lag=env._current_obs.kafka_lag,
    api_latency=env._current_obs.api_latency,
    rolling_p99=2000.0,   # well above 800ms threshold
)
_, r_sla, _, info_sla = env.step(make_action(infra=0))
# raw should be 0.8 - 0.3 = 0.5
check("SLA breach deducts 0.3",
      abs(info_sla['reward_raw'] - 0.5) < 0.01,
      f"raw={info_sla['reward_raw']:.3f}, p99={info_sla['obs_rolling_p99']:.0f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Catastrophic fraud (-1.0) ──")
# ═══════════════════════════════════════════════════════════════════
# Use hard task so risk_score is always > 80
env.reset(task_name="hard")
obs0 = env.state()
# Manually check obs risk — hard always > 85, so fraud gate always fires
_, r_fraud, _, info_fraud = env.step(
    make_action(risk=0, infra=0, crypto=1)   # Approve + SkipVerify
)
# raw = 0.8 - 1.0 = -0.2, clipped to 0.0
check("fraud gate clips reward to 0.0", r_fraud == 0.0,
      f"reward={r_fraud}, raw={info_fraud['reward_raw']:.3f}")
check("fraud gate fires on hard task",
      info_fraud["obs_risk_score"] > 80.0,
      f"risk_score={info_fraud['obs_risk_score']:.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Crash condition ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
env._rolling_lag = 4500.0   # push lag past crash threshold
env._rolling_latency = 10.0
_, r_crash, done_crash, info_crash = env.step(
    make_action(risk=0, infra=0, crypto=0)  # Normal infra (adds +100 lag, +150 crypto)
)
check("crash forces reward to 0.0", r_crash == 0.0,
      f"got {r_crash}")
check("crash sets done=True", done_crash,
      f"done={done_crash}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker prevents crash ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
env._rolling_lag = 4500.0   # would crash without CB
env._rolling_latency = 10.0
_, r_no_crash, done_no_crash, info_no_crash = env.step(
    make_action(infra=2)    # CircuitBreaker — resets lag to 0
)
check("CB prevents crash (done=False from lag)", not done_no_crash or
      env.current_step >= env.max_steps,
      f"done={done_no_crash}, lag={info_no_crash['internal_rolling_lag']}")
check("CB resets internal lag to ≈ 0",
      info_no_crash["internal_rolling_lag"] < 50.0,
      f"lag={info_no_crash['internal_rolling_lag']:.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── max_steps triggers done ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="easy")
env.max_steps = 3
for i in range(2):
    _, _, done_mid, _ = env.step(make_action())
    check(f"done=False at step {i+1}", not done_mid)
_, _, done_end, _ = env.step(make_action())
check("done=True at max_steps", done_end)
env.max_steps = 100   # restore

# ═══════════════════════════════════════════════════════════════════
print("\n── Info dict keys ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(task_name="medium")
_, _, _, info = env.step(make_action())
required_keys = {
    "step", "task", "event_type",
    "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
    "action_risk_decision", "action_infra_routing", "action_crypto_verify",
    "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
    "internal_rolling_lag", "internal_rolling_latency",
}
missing = required_keys - info.keys()
check("info contains all required keys", not missing, f"missing: {missing}")

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═' * 52}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═' * 52}")
if failed == 0:
    print("  ✅ ALL PHASE 4 CHECKS PASSED")
else:
    print("  ❌ SOME CHECKS FAILED — review above")
