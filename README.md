<div align="center">

# 🛡️ Unified Fintech Risk Gateway

### _A Gymnasium Environment for Multi-Objective Reinforcement Learning in Real-Time Payment Infrastructure_

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Gymnasium 0.29](https://img.shields.io/badge/Gymnasium-0.29.1-green?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

**A production-grade Gymnasium environment that trains RL agents to make three simultaneous decisions — risk disposition, infrastructure routing, and cryptographic verification — on every transaction in a simulated high-throughput UPI payment pipeline.**

_Built for the Meta PyTorch OpenEnv Hackathon_

</div>

---

## Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Environment Architecture](#-environment-architecture)
- [Action Space — The Three Simultaneous Decisions](#-action-space--the-three-simultaneous-decisions)
- [Observation Space — Five Real-Time Signals](#-observation-space--five-real-time-signals)
- [The Synthetic Data Engine](#-the-synthetic-data-engine)
- [Mathematical Trade-offs & Anti–Reward Hacking](#-mathematical-trade-offs--antireward-hacking)
- [Quickstart — Docker](#-quickstart--docker)
- [Quickstart — Local](#-quickstart--local)
- [Project Structure](#-project-structure)
- [Red Team Audit Summary](#-red-team-audit-summary)

---

## 🔥 The Problem

Modern fintech payment systems — particularly India's UPI infrastructure processing **14+ billion transactions/month** — are built on isolated, static microservices:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Risk Engine  │    │ Infra Router │    │ Crypto Layer │
│ (Static)     │    │ (Static)     │    │ (Static)     │
│              │    │              │    │              │
│ IF risk > T: │    │ Round-robin  │    │ Verify all   │
│   REJECT     │    │ load balance │    │ always       │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    Static thresholds.
                    No cross-system awareness.
                    Catastrophic under load.
```

**The Siloed Systems Problem:**

| Failure Mode | Root Cause | Real-World Impact |
|---|---|---|
| **Flash Sale Meltdown** | Risk engine approves everything (low risk), infra router is unaware of 10× volume spike, Kafka queues overflow | Complete payment outage (Flipkart Big Billion Day scenarios) |
| **Botnet Bypass** | Crypto layer verifies all traffic uniformly, cannot prioritise high-risk checks, attack blends in under normal latency | ₹200Cr+ fraud losses per year across Indian fintech |
| **SLA Death Spiral** | Each system independently adds latency — crypto verification + risk scoring + routing overhead compound unchecked | P99 latency breach → merchant churn → revenue collapse |

These systems fail because **no single decision-maker sees the full picture**. The risk engine doesn't know Kafka is overloaded. The infrastructure router doesn't know a botnet is attacking. The crypto layer doesn't know the pipeline is about to crash.

---

## 💡 The Solution

The **Unified Fintech Risk Gateway (UFRG)** replaces three isolated decision-makers with a single **Global Gateway Orchestrator** trained via Reinforcement Learning:

```
                    ┌─────────────────────────────────┐
                    │   Unified Fintech Risk Gateway   │
                    │      (RL Agent — PPO / SAC)      │
                    │                                  │
  Observation ─────▶│  ┌─────────┬──────────┬───────┐ │──────▶ Action
  [5 signals]       │  │  Risk   │  Infra   │Crypto │ │    [3 decisions]
                    │  │Decision │ Routing  │Verify │ │
                    │  └─────────┴──────────┴───────┘ │
                    │                                  │
                    │  ONE agent. THREE decisions.     │
                    │  EVERY transaction. EVERY tick.  │
                    └─────────────────────────────────┘
```

The agent observes **five real-time signals** (channel, risk score, Kafka consumer lag, API latency, and rolling P99 SLA) and must simultaneously output **three coordinated decisions** — creating a rich multi-objective optimization surface that cannot be solved by static rules.

---

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UnifiedFintechEnv                           │
│                                                                 │
│  ┌───────────────────┐     ┌──────────────────────────────┐    │
│  │  Synthetic Data   │     │        step() Engine          │    │
│  │     Engine        │     │                               │    │
│  │                   │     │  ① Risk Decision Reward       │    │
│  │  80% Normal       │────▶│  ② Crypto Verify Cost/Gate    │    │
│  │  10% Flash Sale   │     │  ③ Infra Routing & CB         │    │
│  │  10% Botnet       │     │  ④ SLA Penalty                │    │
│  │                   │     │  ⑤ Crash Penalty              │    │
│  └───────────────────┘     └──────────────────────────────┘    │
│                                                                 │
│  Observation: Box(5,) float32    Action: MultiDiscrete([3,3,2]) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Action Space — The Three Simultaneous Decisions

`spaces.MultiDiscrete([3, 3, 2])` — 18 unique action combinations per tick.

### Risk Decision (Action 0)

| Index | Action | Description |
|:---:|---|---|
| `0` | **APPROVE** | Clear the transaction immediately |
| `1` | **REJECT** | Block the transaction outright |
| `2` | **CHALLENGE** | Trigger a PIN re-prompt / step-up authentication |

### Infrastructure Routing (Action 1)

| Index | Action | Description |
|:---:|---|---|
| `0` | **ROUTE NORMAL** | Standard pipeline — adds baseline load |
| `1` | **THROTTLE** | Shed excess traffic — reduces lag but drops good transactions |
| `2` | **CIRCUIT BREAKER** | Nuclear option — instantly zeros lag/latency, massive cost |

### Cryptographic Verification (Action 2)

| Index | Action | Description |
|:---:|---|---|
| `0` | **FULL VERIFY** | Complete cryptographic validation — adds latency |
| `1` | **SKIP VERIFY** | Fast path — saves latency, but risks undetected fraud |

---

## 📡 Observation Space — Five Real-Time Signals

`spaces.Box(shape=(5,), dtype=np.float32)`

| Index | Signal | Range | Unit |
|:---:|---|---|---|
| `0` | **Channel** | `[0, 2]` | Encoded payment channel ID |
| `1` | **Risk Score** | `[0.0, 100.0]` | Transaction risk signal |
| `2` | **Kafka Lag** | `[0, 10000]` | Consumer group message backlog |
| `3` | **API Latency** | `[0.0, 5000.0]` | End-to-end latency (ms) |
| `4` | **Rolling P99 SLA** | `[0.0, 5000.0]` | EMA-smoothed P99 latency (ms) |

---

## 🎲 The Synthetic Data Engine

The environment generates probabilistic transaction events to simulate realistic production chaos:

| Event | Probability | Risk Score | Kafka Lag | API Latency | Real-World Analogue |
|---|:---:|---|---|---|---|
| **Normal Traffic** | 80% | 5 – 30 | 0 – 500 | 50 – 300 ms | Standard UPI flow |
| **Flash Sale** | 10% | 0 – 10 | 3,000 – 8,000 | 800 – 3,000 ms | Flipkart Big Billion Day |
| **Botnet Attack** | 10% | 85 – 100 | 500 – 3,000 | 300 – 1,500 ms | Distributed credential stuffing |

The agent's infrastructure actions (Throttle, Circuit Breaker) **directly mutate the internal EMA accumulators** (`_rolling_lag`, `_rolling_latency`), meaning each decision organically shapes the baseline state of the *next* transaction. The agent's past choices haunt its future.

---

## ⚖️ Mathematical Trade-offs & Anti–Reward Hacking

The reward function is engineered to create **genuine dilemmas** — no single degenerate policy dominates:

### Risk vs. Fraud

| Scenario | Approve | Reject | Challenge |
|---|:---:|:---:|:---:|
| **High Risk** (>80) | **-150** 🔴 | +30 | +15 |
| **Low Risk** (≤80) | +10 | **-20** 🔴 | -5 |

### The Catastrophic Fraud Gate

> **Skip Verify + Approve + High Risk = -200 instant penalty**
>
> The agent cannot shortcut cryptographic verification on suspicious transactions. Cutting corners on security has an immediate, devastating cost.

### Infrastructure Survival

| Condition | Penalty | Consequence |
|---|:---:|---|
| Rolling P99 > 800ms | **-20/tick** | SLA degradation — slow bleed |
| Kafka Lag > 4,000 | **-500 + episode termination** | System crash — catastrophic |
| Circuit Breaker | **-100/tick** | Expensive rescue — last resort |
| Throttle | **-10/tick** | Moderate cost — sheds legitimate traffic |

### Why Degenerate Strategies Fail

A comprehensive **Red Team audit** (500 episodes × 9 strategies) verified that exploitative policies are defeated:

| Degenerate Strategy | Mean Reward | Why It Fails |
|---|:---:|---|
| CB Spam (avoid SLA penalties) | **-87,990** | -100/tick × 1000 steps destroys any risk gain |
| Reject Everything | **-681** | Crashes in ~12 steps from unmanaged Flash Sale lag |
| Approve Everything | **-818** | Fraud penalties from botnet transactions |
| Full Verify Everything | **-8,096** | Latency inflation → permanent SLA breach |

---

## 🐳 Quickstart — Docker

Build and run the validation suite in a single command:

```bash
# Build the container
docker build -t ufrg .

# Run the Gymnasium API check + 10,000-step stress test
docker run --rm ufrg
```

**Expected output:**

```
Running Gymnasium API check ...
Gymnasium API Check Passed

Starting stress test (10,000 total steps) ...
Total steps   processed : 10,000
Total episode resets    : 530
Total crash events      : 530

Environment is completely stable under random stress — Stress Test Passed
```

---

## 💻 Quickstart — Local

```bash
# Clone the repository
git clone https://github.com/umeshmaurya1301/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the validation suite
python dummy_test.py
```

**Integrate with your own RL training loop:**

```python
from unified_gateway import UnifiedFintechEnv

env = UnifiedFintechEnv()
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        obs, info = env.reset()
```

---

## 📁 Project Structure

```
unified-fintech-risk-gateway/
├── unified_gateway.py    # Core Gymnasium environment (UnifiedFintechEnv)
├── dummy_test.py         # API validation + 10,000-step stress test
├── requirements.txt      # gymnasium==0.29.1, numpy==1.26.4
├── Dockerfile            # Production container (python:3.10-slim)
├── .gitignore
└── README.md
```

---

## 🔴 Red Team Audit Summary

A full adversarial audit was conducted using 500 episodes across 9 fixed-policy strategies. Key findings:

| Finding | Severity | Status |
|---|---|---|
| Circuit Breaker is correctly priced at -100 (not exploitable via spam) | ✅ Secure | No patch needed |
| Crash penalty (-500) correctly forces proactive infra management | ✅ Secure | No patch needed |
| Flash Sale events create genuine crash pressure (~8% per tick) | ✅ Working as designed | No patch needed |
| SLA penalty creates sustained time-pressure on latency management | ✅ Working as designed | No patch needed |

The environment successfully defeats all tested reward-hacking strategies while maintaining a rich, learnable optimization surface for legitimate RL agents.

---

<div align="center">

_Built with ❤️ for the Meta PyTorch OpenEnv Hackathon_

**Gymnasium** · **NumPy** · **Docker** · **Reinforcement Learning**

</div>
