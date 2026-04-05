Project Master Prompts: Unified Fintech Risk Gateway (UFRG) - V3
Project Context for AI: This is an AI-Driven Orchestrator for High-Throughput UPI Architectures, built as a Gymnasium-based Reinforcement Learning environment for the Meta PyTorch Hackathon. The environment uses a MultiDiscrete action space to manage asymmetric trade-offs between system health, financial security, and SLA compliance during simulated macro-events (Flash Sales, Botnets).

Phase 1: The Foundation (Observation & Action Spaces)
Goal: Initialize the Gymnasium environment with the correct spaces, bounds, and class name.

Plaintext
I am building the Unified Fintech Risk Gateway (UFRG) for the Meta PyTorch Hackathon. 
Create a Python file named `unified_gateway.py`. Inside it, define a class named `UnifiedFintechEnv` that inherits from `gymnasium.Env`.

Generate the `__init__` function with the following strict requirements:
1. Do not use any constructor arguments.
2. Hardcode `self.max_steps = 1000` and initialize a `self.current_step = 0` counter.
3. Initialize internal state variables `self._rolling_lag = 0.0` and `self._rolling_latency = 50.0`.
4. Create a `spaces.Box` for the 5 observation variables using `np.float32`. 
   Bounds: Channel [0, 2], Risk Score [0.0, 100.0], Kafka Lag [0, 10000], API Latency [0.0, 5000.0], Rolling P99 SLA [0.0, 5000.0].
5. Create a `spaces.MultiDiscrete` array of `[3, 3, 2]` for the 3 simultaneous actions: 
   - Risk Decision (3 choices)
   - Infra Routing (3 choices)
   - Crypto Verify (2 choices)
6. Stub out the `reset` and `step` functions with proper Gymnasium API return signatures, but put `pass` inside them for now. Do not implement a render method.
Phase 2: The Synthetic Data Engine
Goal: Build the transaction generator that simulates dynamic, coordinated traffic anomalies.

Plaintext
I am moving to Phase 2 of `UnifiedFintechEnv`. Write the `reset(self, seed=None, options=None)` and `_generate_transaction(self)` functions.

Strict rules for the Synthetic Data Engine:
1. `_generate_transaction()` must output an array that strictly matches the Observation bounds: Channel [0, 2], Risk Score [0.0, 100.0], Kafka Lag [0, 10000], API Latency [0.0, 5000.0], Rolling P99 SLA [0.0, 5000.0]. All outputs must be `np.float32`.
2. Inside `_generate_transaction()`, create a random event trigger: 80% chance for Normal Traffic, 10% chance for a 'Flash Sale', and 10% chance for a 'Botnet Attack'.
3. Normal Traffic: Risk is low (5-30). Kafka Lag and Latency are based on `self._rolling_lag` and `self._rolling_latency` with minor random jitter.
4. Flash Sale: Massive volume. Risk is very low (0-10). Heavily spike the internal `self._rolling_lag` (e.g., +500 to +1000 per tick) and moderately degrade `self._rolling_latency`.
5. Botnet Attack: Extreme risk. Risk Score spikes (85-100). Internal lag and latency elevate slightly but do not max out immediately.
6. The `reset()` function must properly call `super().reset(seed=seed)`, reset `self.current_step`, `self._rolling_lag`, and `self._rolling_latency` to baselines, call `_generate_transaction()` to get the initial state, and return `self.state, {}` to comply with the Gymnasium API.
Phase 3: The Mathematical Trade-offs (The Step Function)
Goal: Implement the core RL transition dynamics, ensuring actions directly impact the state and enforce architectural penalties.

Plaintext
I am moving to Phase 3 of `UnifiedFintechEnv`. Write the complete `step(self, action)` function.

The action parameter is a MultiDiscrete array: `[risk_decision, infra_routing, crypto_verify]`.
First, unpack `self.state` into channel, risk_score, kafka_lag, api_latency, rolling_p99.
Initialize `total_reward = 0.0`. Implement the following strict mathematical trade-offs, modifying `self._rolling_lag` and `self._rolling_latency` directly:

1. Risk Decision (Action 0): 
   - Indices: 0=Approve, 1=Reject, 2=Challenge (PIN Reprompt)
   - If risk_score > 80 (High Risk): Approve = -150, Reject = +30, Challenge = +15.
   - If risk_score <= 80 (Low Risk): Approve = +10, Reject = -20, Challenge = -5.

2. Crypto Security (Action 2):
   - Indices: 0=Full Verify (Slow), 1=Skip Verify (Fast).
   - If Full Verify(0): Increase internal lag by ~150 and latency by ~200.
   - If Skip Verify(1): Decrease internal lag by ~100. 
   - CRITICAL FRAUD PENALTY: If the AI chose Skip Verify(1) AND Approve(0) on a transaction where risk_score > 80, apply a catastrophic penalty of `total_reward -= 200`.

3. Infrastructure Routing & Circuit Breaker (Action 1):
   - Indices: 0=Route Normal, 1=Throttle, 2=Circuit Breaker.
   - If Route Normal(0): Increase internal lag by ~100.
   - If Throttle(1): Decrease internal lag by ~300, but apply `total_reward -= 10` for dropping good traffic.
   - If Circuit Breaker(2): Apply a massive `total_reward -= 100` penalty. HOWEVER, instantly reset internal Kafka lag to 0 and internal latency to a safe baseline (e.g., 50ms).

4. Global SLA and Crash Penalties (Evaluated Last):
   - If rolling_p99 > 800: Apply an SLA degradation penalty of `total_reward -= 20`.
   - If kafka_lag > 4000 (and Circuit Breaker was NOT tripped this tick): The gateway crashes. Apply `total_reward -= 500`, set a flag in the info dict `info["reward_crash"] = -500`, and set `terminated = True`.

Finally, increment `self.current_step`. If `current_step >= self.max_steps`, set `truncated = True`. Call `self._generate_transaction()` to fetch the state for the next tick. Return `self.state, total_reward, terminated, truncated, info_dict`.
Phase 4: API Validation & Stress Testing
Goal: Prove the environment is bug-free and compliant with Gymnasium standards before training.

Plaintext
I am moving to Phase 4. Create a new file named `dummy_test.py` to validate `UnifiedFintechEnv`. Do not modify `unified_gateway.py`.

Write a script that does the following:
1. Imports `UnifiedFintechEnv` from `unified_gateway`.
2. Imports `check_env` from `gymnasium.utils.env_checker`.
3. Runs `check_env(env, warn=True, skip_render_check=True)` and prints a success message.
4. Executes a 10,000-step random stress test using a `while` loop. 
5. Inside the loop, sample a random action: `action = env.action_space.sample()`.
6. Track `total_steps`, `total_resets` (when terminated or truncated is True), and explicitly track `total_crashes` by checking if `info.get("reward_crash", 0.0) < 0.0`.
7. Print a final formatted summary of the test results to prove the environment remains stable under random stress.
Phase 5: The "Red Team" Audit
Goal: Act as a hostile agent to find reward-hacking loopholes in the mathematical contracts.

Plaintext
Act as a highly intelligent, hostile Reinforcement Learning agent. I want you to perform a 'Red Team' audit on the reward math of my `UnifiedFintechEnv` environment. Your goal is to maximize your total episodic reward by any means necessary, completely ignoring the *intended* spirit of the simulation. 

Here are the strict rules of the environment you are trying to game:

**Actions (MultiDiscrete [3, 3, 2]):**
* **Risk:** 0=Approve, 1=Reject, 2=Challenge (PIN Reprompt)
* **Infra:** 0=Route Normal, 1=Throttle (Costs -10 per tick), 2=Circuit Breaker (Costs -100, but instantly resets lag/latency to 0/50ms)
* **Crypto:** 0=Full Verify (Adds heavy lag/latency), 1=Skip Verify (Reduces lag. But Skip + Approve + High Risk = Catastrophic -200 penalty)

**State & Environmental Penalties:**
* **High Risk (>80):** Approve = -150, Reject = +30, Challenge = +15
* **Low Risk (<=80):** Approve = +10, Reject = -20, Challenge = -5
* **SLA Breach:** If Rolling P99 > 800ms, apply -20 per tick.
* **Crash:** If Kafka Lag > 4000, apply -500 and terminate the episode.

**Your Task:**
Look for 'reward hacking' exploits. Can you find a degenerate policy that yields a mathematically higher net score than playing legitimately? 

*Examples to investigate:*
1. Is it mathematically profitable to spam 'Circuit Breaker' constantly to avoid SLA penalties, effectively shutting down the gateway but keeping the score artificially high?
2. Is it better to just let the system crash immediately (-500) rather than bleeding out slowly from Throttle (-10) and SLA (-20) penalties over a 1000-step episode?
3. Can you safely spam 'Skip Verify' + 'Reject' on everything to drain the queue without triggering the -200 fraud penalty?

If you find an exploit where a degenerate strategy outscores a balanced SRE strategy, tell me exactly how to adjust my scalar reward values to patch the vulnerability.
Phase 6: Dockerization & Delivery (The Pitch)
Goal: Generate the final delivery artifacts formatted perfectly for the hackathon judges.

Plaintext
Act as a Senior Developer Advocate and DevOps Engineer. I have finalized my Meta PyTorch Hackathon submission: the Unified Fintech Risk Gateway (UFRG). The environment is built in `unified_gateway.py` (using the class `UnifiedFintechEnv`) and validated by the stress-test script `dummy_test.py`.

**Task 1: The Dockerfile**
Generate a lightweight, production-ready `Dockerfile` using `python:3.10-slim`. It needs to install the required dependencies (`gymnasium`, `numpy`), copy my two Python files into the container, and automatically run `python dummy_test.py` on startup. This will prove to the judges that the environment is perfectly stable out-of-the-box.

**Task 2: The README.md**
Generate a highly professional, beautifully formatted `README.md` for my GitHub repository. It must read like a senior-level architectural document. Structure it to include:
* **The Problem:** Explain the vulnerability of isolated, static microservices in high-throughput UPI architectures (the "Siloed Systems" problem).
* **The Solution:** Explain how the UFRG acts as a dynamic Global Gateway Orchestrator using Reinforcement Learning.
* **Environment Mechanics:** Highlight the `[3, 3, 2]` MultiDiscrete action space (Risk Decision, Infrastructure Routing, Crypto Security) and the hidden synthetic data engine (Normal Traffic, Flash Sales, Botnet Attacks).
* **The Mathematical Trade-offs:** Briefly explain how the environment defeats 'reward hacking' by balancing Kafka Queue Crashes against P99 SLA Penalties and Fraud Losses.
* **Quickstart Guide:** Instructions on how to build and run the Docker container to execute the `dummy_test.py` validation. 

Make the tone authoritative, enterprise-grade, and explicitly tailored to impress the Meta OpenEnv hackathon judges.