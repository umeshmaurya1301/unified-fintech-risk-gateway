"""
Phase 4 — Gymnasium API Validation & Stress Test
=================================================
Validates that UnifiedFintechEnv is fully compliant with the Gymnasium API
and remains stable over 10 000 random steps.
"""

from unified_gateway import UnifiedFintechEnv
from gymnasium.utils.env_checker import check_env

# ─────────────────────────────────────────────────────────────────────────────
# 1. Official Gymnasium API compliance check
# ─────────────────────────────────────────────────────────────────────────────
print("Running Gymnasium API check ...")
env = UnifiedFintechEnv()
check_env(env, warn=True, skip_render_check=True)
print("Gymnasium API Check Passed\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Random-action stress test — 10 000 total steps
# ─────────────────────────────────────────────────────────────────────────────
TARGET_STEPS = 10_000

env = UnifiedFintechEnv()
env.reset(seed=0)

total_steps   = 0
total_resets  = 0
total_crashes = 0

print(f"Starting stress test ({TARGET_STEPS:,} total steps) ...")

while total_steps < TARGET_STEPS:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    if info.get("reward_crash", 0.0) < 0.0:
        total_crashes += 1

    if terminated or truncated:
        env.reset()
        total_resets += 1

print(f"Total steps   processed : {total_steps:,}")
print(f"Total episode resets    : {total_resets:,}")
print(f"Total crash events      : {total_crashes:,}")
print()
print("Environment is completely stable under random stress — Stress Test Passed")
