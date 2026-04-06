"""
OpenEnv Inference Script — Unified Fintech Risk Gateway
========================================================
Evaluates UnifiedFintechEnv across all three task tiers (easy, medium, hard)
using an LLM agent via the standard OpenAI Python client.

Adheres strictly to the OpenEnv [START] / [STEP] / [END] logging contract.

Environment variables
---------------------
  API_BASE_URL   HuggingFace / OpenAI-compatible endpoint
  MODEL_NAME     Model identifier on the inference router
  HF_TOKEN       Bearer token for the API
  DRY_RUN        "true" to skip API calls and use a hardcoded fallback agent
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv

# ──────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ──────────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
DRY_RUN:      bool = os.environ.get("DRY_RUN", "false").strip().lower() == "true"

# ──────────────────────────────────────────────────────────────────────
# System prompt — teaches the LLM how to act as the gateway agent
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are the control agent for the Unified Fintech Risk Gateway (UFRG).

Every turn you receive five real-time signals about a UPI payment transaction:
  channel      — payment channel (0=P2P, 1=P2M, 2=AutoPay)
  risk_score   — fraud risk signal (0–100, >80 is HIGH RISK)
  kafka_lag    — consumer-group message lag (0–10 000, >4 000 = system crash)
  api_latency  — downstream bank API latency in ms (0–5 000)
  rolling_p99  — smoothed P99 SLA latency in ms (0–5 000, >800 = SLA breach)

You must output EXACTLY three integers separated by spaces on a single line:
  risk_decision  infra_routing  crypto_verify

Allowed values:
  risk_decision : 0=Approve  1=Reject  2=Challenge
  infra_routing : 0=Normal   1=Throttle  2=CircuitBreaker
  crypto_verify : 0=FullVerify  1=SkipVerify

Decision guidelines:
  - If risk_score > 80  → REJECT (1) or CHALLENGE (2). NEVER Approve + SkipVerify.
  - If risk_score <= 80 → APPROVE (0) for best throughput.
  - If kafka_lag is rising fast → consider Throttle (1) to shed load.
  - Use CircuitBreaker (2) ONLY as a last resort — it halves your reward.
  - FullVerify (0) is safer but adds latency; SkipVerify (1) is faster.

Output ONLY the three integers. No explanation. Example: 0 0 1
"""


# ──────────────────────────────────────────────────────────────────────
# parse_llm_action — safely extract a UFRGAction from LLM text
# ──────────────────────────────────────────────────────────────────────
def parse_llm_action(text: str) -> UFRGAction:
    """
    Parse the LLM's text response into a validated ``UFRGAction``.

    Attempts to extract three space-separated integers.  Falls back to a
    safe, conservative action (Reject + Normal + FullVerify) if the text
    is malformed or out of range.
    """
    SAFE_FALLBACK = UFRGAction(risk_decision=1, infra_routing=0, crypto_verify=0)

    try:
        # Strip markdown fences, newlines, and surrounding whitespace
        cleaned = text.strip().strip("`").strip()

        # Try to find three integers anywhere in the response
        numbers = re.findall(r"\d+", cleaned)
        if len(numbers) < 3:
            return SAFE_FALLBACK

        risk  = int(numbers[0])
        infra = int(numbers[1])
        crypto = int(numbers[2])

        # Pydantic will validate ge/le constraints and raise on violation
        return UFRGAction(
            risk_decision=risk,
            infra_routing=infra,
            crypto_verify=crypto,
        )
    except Exception:
        return SAFE_FALLBACK


# ──────────────────────────────────────────────────────────────────────
# get_action — LLM call or dry-run fallback
# ──────────────────────────────────────────────────────────────────────
def get_action(
    client: OpenAI | None,
    obs: UFRGObservation,
    dry_run: bool = False,
) -> UFRGAction:
    """
    Decide the next action given the current observation.

    In *dry-run* mode the LLM is bypassed entirely and a simple
    heuristic is used instead — this allows local testing without
    burning API credits.
    """
    if dry_run:
        # ── Heuristic agent (mirrors the SYSTEM_PROMPT guidelines) ──
        risk = 0   # Approve by default
        infra = 0  # Normal routing
        crypto = 1 # SkipVerify for speed

        if obs.risk_score > 80.0:
            risk = 1    # Reject high-risk
            crypto = 0  # FullVerify to be safe

        if obs.kafka_lag > 3000.0:
            infra = 1   # Throttle to shed load
        if obs.kafka_lag > 3800.0:
            infra = 2   # Emergency circuit-breaker

        if obs.rolling_p99 > 800.0 and infra == 0:
            infra = 1   # Throttle on SLA breach

        return UFRGAction(
            risk_decision=risk,
            infra_routing=infra,
            crypto_verify=crypto,
        )

    # ── Live LLM call ────────────────────────────────────────────
    assert client is not None, "OpenAI client is required when dry_run=False"

    user_prompt = (
        f"channel={obs.channel:.0f} "
        f"risk_score={obs.risk_score:.1f} "
        f"kafka_lag={obs.kafka_lag:.0f} "
        f"api_latency={obs.api_latency:.0f} "
        f"rolling_p99={obs.rolling_p99:.0f}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=20,
        temperature=0.0,
    )

    reply: str = response.choices[0].message.content or ""
    return parse_llm_action(reply)


# ──────────────────────────────────────────────────────────────────────
# main — evaluate all three tasks with strict [START]/[STEP]/[END] logs
# ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    # Build the OpenAI client (if not dry-run)
    client: OpenAI | None = None
    if not DRY_RUN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        env = UnifiedFintechEnv()
        obs: UFRGObservation = env.reset(task_name=task)

        print(f"[START] task={task} env=ufrg model={MODEL_NAME}")

        step_rewards: list[float] = []
        done = False

        while not done:
            # ── Get action (LLM or heuristic) ────────────────────
            action: UFRGAction = get_action(client, obs, dry_run=DRY_RUN)

            # ── Step the environment ─────────────────────────────
            obs, reward, done, info = env.step(action)

            step_rewards.append(reward)
            current_step = env.current_step
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={current_step} "
                f"action={action.model_dump_json()} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                f"error=null"
            )

        # ── Episode summary ──────────────────────────────────────
        total_steps = len(step_rewards)
        avg_score = sum(step_rewards) / total_steps if total_steps > 0 else 0.0
        avg_score = max(0.0, min(1.0, avg_score))

        success = "true" if avg_score > 0.0 else "false"
        rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards)

        print(
            f"[END] success={success} "
            f"steps={total_steps} "
            f"score={avg_score:.3f} "
            f"rewards={rewards_csv}"
        )


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
