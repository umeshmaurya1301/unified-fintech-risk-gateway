"""Quick smoke test for the Foundation (Data Models) layer."""
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv
import numpy as np

# ── 1. UFRGAction: valid construction ────────────────────────────────────
a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
print(f"[PASS] UFRGAction created: {a.model_dump()}")

# ── 2. UFRGAction: boundary validation ───────────────────────────────────
for bad in [
    dict(risk_decision=3, infra_routing=0, crypto_verify=0),
    dict(risk_decision=-1, infra_routing=0, crypto_verify=0),
    dict(risk_decision=0, infra_routing=0, crypto_verify=2),
]:
    try:
        UFRGAction(**bad)
        print(f"[FAIL] Should have rejected {bad}")
    except Exception:
        print(f"[PASS] Correctly rejected {bad}")

# ── 3. UFRGObservation: construction + round-trip ────────────────────────
arr = np.array([1.0, 45.5, 1200.0, 320.0, 400.0], dtype=np.float32)
obs = UFRGObservation.from_array(arr)
print(f"[PASS] UFRGObservation from_array: {obs.model_dump()}")

back = obs.to_array()
assert np.allclose(arr, back), f"Round-trip mismatch: {arr} vs {back}"
print(f"[PASS] to_array round-trip matches")

# ── 4. UnifiedFintechEnv __init__ state ──────────────────────────────────
env = UnifiedFintechEnv()
checks = [
    ("max_steps",        env.max_steps,        100),
    ("current_step",     env.current_step,     0),
    ("_rolling_lag",     env._rolling_lag,     0.0),
    ("_rolling_latency", env._rolling_latency, 50.0),
]
for name, actual, expected in checks:
    assert actual == expected, f"{name}: got {actual}, expected {expected}"
    print(f"[PASS] {name} = {actual}")

print(f"[PASS] action_space      = {env.action_space}")
print(f"[PASS] obs_space.shape   = {env.observation_space.shape}")
print(f"[PASS] obs_space.low     = {env.observation_space.low}")
print(f"[PASS] obs_space.high    = {env.observation_space.high}")

print("\n══ ALL FOUNDATION CHECKS PASSED ══")
