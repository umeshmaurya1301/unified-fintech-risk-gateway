"""
Phase 2 + 3 — Foundation & Task-Driven Reset / State Verification
==================================================================
Validates UFRGAction, UFRGObservation, reset(task_name), state(),
and _generate_transaction() across all three difficulty tiers.
"""
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  — {detail}")


# ═══════════════════════════════════════════════════════════════════════
print("── Phase 2: Pydantic Models ──")
# ═══════════════════════════════════════════════════════════════════════

a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
check("UFRGAction valid construction", a.risk_decision == 1)

for bad, key in [
    (dict(risk_decision=3, infra_routing=0, crypto_verify=0), "risk_decision=3"),
    (dict(risk_decision=0, infra_routing=0, crypto_verify=2), "crypto_verify=2"),
    (dict(risk_decision=-1, infra_routing=0, crypto_verify=0), "risk_decision=-1"),
]:
    try:
        UFRGAction(**bad)
        check(f"Reject {key}", False, "no error raised")
    except Exception:
        check(f"Reject {key}", True)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: reset() ──")
# ═══════════════════════════════════════════════════════════════════════

env = UnifiedFintechEnv()

for task in ["easy", "medium", "hard"]:
    obs = env.reset(task_name=task)

    check(f"reset('{task}') returns UFRGObservation",
          isinstance(obs, UFRGObservation))
    check(f"reset('{task}') NOT a tuple",
          not isinstance(obs, tuple),
          f"got {type(obs)}")
    check(f"reset('{task}') current_task stored",
          env.current_task == task)
    check(f"reset('{task}') current_step == 0",
          env.current_step == 0)
    if task == "hard":
        # Hard profile inflates _rolling_lag during the first _generate_transaction
        # call inside reset(), so it won't be 0.0 after reset returns.
        check(f"reset('{task}') _rolling_lag in hard-task range",
              0.0 < env._rolling_lag < 500.0,
              f"got {env._rolling_lag}")
    else:
        # Easy starts near 0.0 (only minor jitter from baseline).
        # Medium can spike on the first tick if flash-sale fires (20% chance),
        # so we allow a wider but still bounded range.
        upper = 50.0 if task == "easy" else 1200.0
        check(f"reset('{task}') _rolling_lag in sane range",
              env._rolling_lag < upper,
              f"got {env._rolling_lag}")
    # _rolling_latency starts at 50.0 but EMA update in _generate_transaction
    # will shift it slightly — just check it's in a sane range
    check(f"reset('{task}') _rolling_latency near baseline",
          0.0 < env._rolling_latency < 500.0,
          f"got {env._rolling_latency}")

# ── Default task_name ─────────────────────────────────────────────────
obs_default = env.reset()
check("reset() defaults to 'easy'", env.current_task == "easy")

# ── Bad task_name ─────────────────────────────────────────────────────
try:
    env.reset(task_name="nightmare")
    check("reset('nightmare') raises ValueError", False, "no error raised")
except ValueError:
    check("reset('nightmare') raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: state() ──")
# ═══════════════════════════════════════════════════════════════════════

obs = env.reset(task_name="easy")
st = env.state()
check("state() returns UFRGObservation", isinstance(st, UFRGObservation))
check("state() matches reset() result",
      st.channel == obs.channel
      and st.risk_score == obs.risk_score
      and st.kafka_lag == obs.kafka_lag)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: _generate_transaction() per task ──")
# ═══════════════════════════════════════════════════════════════════════

# EASY: risk should always be low
env.reset(task_name="easy")
easy_risks = []
for _ in range(50):
    obs = env._generate_transaction("easy")
    easy_risks.append(obs.risk_score)
check("easy: all risk_scores ≤ 30",
      all(r <= 30.01 for r in easy_risks),
      f"max was {max(easy_risks):.1f}")
check("easy: all risk_scores ≥ 5",
      all(r >= 4.99 for r in easy_risks),
      f"min was {min(easy_risks):.1f}")

# HARD: risk should always be high
env.reset(task_name="hard")
hard_risks = []
for _ in range(50):
    obs = env._generate_transaction("hard")
    hard_risks.append(obs.risk_score)
check("hard: all risk_scores ≥ 85",
      all(r >= 84.99 for r in hard_risks),
      f"min was {min(hard_risks):.1f}")

# MEDIUM: majority should be normal, some flash-sale
env.reset(task_name="medium")
events = []
for _ in range(200):
    env._generate_transaction("medium")
    events.append(env._last_event_type)
normal_pct = events.count("normal") / len(events) * 100
flash_pct = events.count("flash_sale") / len(events) * 100
check(f"medium: ~80% normal (got {normal_pct:.0f}%)",
      55 < normal_pct < 95)
check(f"medium: ~20% flash_sale (got {flash_pct:.0f}%)",
      5 < flash_pct < 45)

# MEDIUM flash-sale should spike lag
env.reset(task_name="medium")
initial_lag = env._rolling_lag
for _ in range(30):
    env._generate_transaction("medium")
check("medium: _rolling_lag grows over 30 steps",
      env._rolling_lag > initial_lag,
      f"initial={initial_lag:.1f}, final={env._rolling_lag:.1f}")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═' * 50}")
if failed == 0:
    print("  ✅ ALL PHASE 2 + 3 CHECKS PASSED")
else:
    print("  ❌ SOME CHECKS FAILED — review output above")
