"""
Phase 4 — step() Verification
==============================
Tests every reward branch, crash condition, done flag, and return types.
Updated for: M1 (new reset signature), H1 (UFRGReward typed return).
"""
from unified_gateway import UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv

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
env.reset(options={"task": "easy"})
result = env.step(make_action())
check("step returns 4-tuple", len(result) == 4,
      f"got {len(result)}-tuple")
obs, typed_reward, done, info = result
check("obs is UFRGObservation", isinstance(obs, UFRGObservation))
check("typed_reward is UFRGReward", isinstance(typed_reward, UFRGReward))
check("typed_reward.value is float", isinstance(typed_reward.value, float))
check("done is bool", isinstance(done, bool))
check("info is dict", isinstance(info, dict))

# ═══════════════════════════════════════════════════════════════════
print("\n── Reward clipping [0.0, 1.0] ──")
# ═══════════════════════════════════════════════════════════════════
for task in ["easy", "medium", "hard"]:
    env.reset(options={"task": task})
    a = make_action(risk=0, infra=0, crypto=0)
    _, tr, _, _ = env.step(a)
    check(f"reward.value in [0,1] ({task})", 0.0 <= tr.value <= 1.0,
          f"got {tr.value}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Baseline reward (no penalties) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr, _, info = env.step(make_action(risk=0, infra=0, crypto=0))
check("baseline ~0.8 with no penalties", 0.5 <= tr.value <= 0.85,
      f"got {tr.value}, raw={info['reward_raw']}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Throttle penalty ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr1, _, _ = env.step(make_action(infra=0))   # Normal
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr2, _, _ = env.step(make_action(infra=1))   # Throttle
check("throttle reduces reward by ~0.2 (normal traffic)",
      abs((tr1.value - tr2.value) - 0.2) < 0.05,
      f"normal={tr1.value:.3f}, throttle={tr2.value:.3f}, diff={tr1.value-tr2.value:.3f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker penalty (-0.5) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_cb, _, info_cb = env.step(make_action(infra=2))
check("circuit-breaker reduces reward by ~0.5",
      abs(info_cb['reward_raw'] - 0.3) < 0.05,
      f"raw={info_cb['reward_raw']:.3f}")
check("circuit_breaker_tripped flag set", info_cb["circuit_breaker_tripped"])
# internal_rolling_lag is captured right after CB fires (lag reset to 0),
# before _generate_transaction re-adds load — so it must be near zero.
check("_rolling_lag ≈ 0 after CB (via info)",
      info_cb["internal_rolling_lag"] < 50.0,
      f"lag={info_cb['internal_rolling_lag']:.1f}")
check("_rolling_latency ≈ 50.0 after CB",
      abs(info_cb["internal_rolling_latency"] - 50.0) < 15.0,
      f"info shows {info_cb['internal_rolling_latency']:.2f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── SLA breach penalty (-0.3) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 0.0
env._current_obs = UFRGObservation(
    channel=env._current_obs.channel,
    risk_score=env._current_obs.risk_score,
    kafka_lag=env._current_obs.kafka_lag,
    api_latency=env._current_obs.api_latency,
    rolling_p99=2000.0,
)
_, tr_sla, _, info_sla = env.step(make_action(infra=0))
check("SLA breach deducts 0.3",
      abs(info_sla['reward_raw'] - 0.5) < 0.01,
      f"raw={info_sla['reward_raw']:.3f}, p99={info_sla['obs_rolling_p99']:.0f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Catastrophic fraud (-1.0) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "hard"})
_, tr_fraud, _, info_fraud = env.step(
    make_action(risk=0, infra=0, crypto=1)   # Approve + SkipVerify
)
check("fraud gate clips reward to 0.0", tr_fraud.value == 0.0,
      f"reward={tr_fraud.value}, raw={info_fraud['reward_raw']:.3f}")
check("fraud gate fires on hard task",
      info_fraud["obs_risk_score"] > 80.0,
      f"risk_score={info_fraud['obs_risk_score']:.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Challenge bonus on high-risk ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "hard"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_rej, _, _ = env.step(make_action(risk=1, infra=0, crypto=0))   # Reject
env.reset(options={"task": "hard"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_chal, _, _ = env.step(make_action(risk=2, infra=0, crypto=0))  # Challenge
check("Challenge earns more than Reject on high-risk",
      tr_chal.value > tr_rej.value,
      f"challenge={tr_chal.value:.3f}, reject={tr_rej.value:.3f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Kafka lag proximity warning ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 3500.0
env._rolling_latency = 10.0
_, tr_warn, _, _ = env.step(make_action(risk=0, infra=0, crypto=1))
check("lag proximity warning appears in breakdown",
      "lag_proximity_warning" in tr_warn.breakdown,
      f"breakdown={tr_warn.breakdown}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Crash condition ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 4500.0
env._rolling_latency = 10.0
_, tr_crash, done_crash, info_crash = env.step(
    make_action(risk=0, infra=0, crypto=0)
)
check("crash forces reward to 0.0", tr_crash.value == 0.0,
      f"got {tr_crash.value}")
check("crash sets done=True", done_crash,
      f"done={done_crash}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker prevents crash ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 4500.0
env._rolling_latency = 10.0
_, tr_no_crash, done_no_crash, info_no_crash = env.step(
    make_action(infra=2)
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
env.reset(options={"task": "easy"})
env.max_steps = 3
for i in range(2):
    _, _, done_mid, _ = env.step(make_action())
    check(f"done=False at step {i+1}", not done_mid)
_, _, done_end, _ = env.step(make_action())
check("done=True at max_steps", done_end)
env.max_steps = 100

# ═══════════════════════════════════════════════════════════════════
print("\n── Info dict keys ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "medium"})
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
