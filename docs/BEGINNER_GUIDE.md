# Unified Fintech Risk Gateway — Complete Beginner's Guide

> **Who is this guide for?**
> This guide is written for someone who is completely new to this project and possibly new to software development concepts like APIs, reinforcement learning, and payment systems. We'll explain everything from scratch — no prior knowledge assumed.

---

## Table of Contents

1. [What is This Project?](#1-what-is-this-project)
2. [Real-World Problem It Solves](#2-real-world-problem-it-solves)
3. [Key Concepts You Need to Know](#3-key-concepts-you-need-to-know)
4. [How the Project Works — Step by Step](#4-how-the-project-works--step-by-step)
5. [Project Structure — What Each Folder Does](#5-project-structure--what-each-folder-does)
6. [The Three Task Levels](#6-the-three-task-levels)
7. [How Scoring (Rewards) Work](#7-how-scoring-rewards-work)
8. [Running the Project — Prerequisites](#8-running-the-project--prerequisites)
9. [Running on Windows](#9-running-on-windows)
10. [Running on macOS](#10-running-on-macos)
11. [Running on Linux](#11-running-on-linux)
12. [Running with Docker (Recommended for All Platforms)](#12-running-with-docker-recommended-for-all-platforms)
13. [Testing the Project](#13-testing-the-project)
14. [Using the Live API](#14-using-the-live-api)
15. [Common Errors and Fixes](#15-common-errors-and-fixes)
16. [Glossary](#16-glossary)

---

## 1. What is This Project?

**Unified Fintech Risk Gateway (UFRG)** is a **simulation environment** where an AI agent learns to manage a real-time payment system.

Think of it like a **video game** for an AI:
- The AI is the **player**
- The payment system is the **game world**
- Every UPI transaction is a new **challenge**
- The AI earns **points (rewards)** for making good decisions and loses points for bad ones

The AI needs to make three decisions for every single payment:
1. **Should we approve, reject, or challenge this payment?** (fraud check)
2. **Should the server work normally, slow down, or stop?** (infrastructure health)
3. **Should we do full security verification or skip it?** (speed vs. security)

These three decisions happen **simultaneously** 100 times per episode (round). The AI that makes the best average decisions wins.

---

## 2. Real-World Problem It Solves

In India, **UPI (Unified Payments Interface)** processes billions of payments every month — from buying chai to paying rent. Behind every payment, a team of engineers (called **SREs — Site Reliability Engineers**) has to:

- **Detect fraud** in real time (scammers, bots, fake transactions)
- **Keep servers healthy** (too many requests can crash the queue system)
- **Meet speed guarantees** (payments must complete within a time limit, called SLA)

These three goals often **conflict with each other**:
- Blocking all suspicious payments = less fraud but slower service
- Speeding up all payments = faster but higher fraud risk
- Throttling the server = safer infrastructure but legitimate users get blocked

A human SRE has to balance all three simultaneously under pressure. This project trains an **AI to do that automatically**.

---

## 3. Key Concepts You Need to Know

### What is UPI?
UPI stands for **Unified Payments Interface**. It's India's digital payment system that lets you send money instantly using your phone (like Google Pay, PhonePe, Paytm).

### What is Kafka?
**Apache Kafka** is a system that handles a queue of messages (like a conveyor belt). In payment systems, every transaction is a "message" put on this belt. If messages pile up faster than they're processed, the belt jams — this is called **Kafka lag**. If lag gets too high (>4000 in this project), the system crashes.

### What is P99 Latency?
**Latency** = how long it takes for a payment to be processed. **P99 latency** means "the slowest 1% of payments." If P99 > 800ms (0.8 seconds), the system is breaching its speed guarantee (SLA). Think of it as: 99% of customers got their payment processed fine, but 1% had to wait too long.

### What is Reinforcement Learning?
**Reinforcement Learning (RL)** is how AI learns by trial and error — like training a dog with treats. The AI tries an action, gets a reward or penalty, and learns to do better next time. Over thousands of episodes, it figures out the best strategy.

### What is a Gymnasium Environment?
**Gymnasium** (formerly OpenAI Gym) is a standard Python library for building RL environments. It defines how to:
- Get the current situation (`observation`)
- Take an action (`step`)
- Get a score (`reward`)
- Know when the episode is over (`done`)

This project builds a Gymnasium environment around the UPI payment simulation.

### What is OpenEnv?
**OpenEnv** is Meta's framework for creating standard evaluation environments for LLM (Large Language Model) agents — like GPT-4, Claude, etc. This project is submitted to an OpenEnv hackathon, meaning the environment must follow OpenEnv's rules so automated judges can test it.

### What is a REST API?
A **REST API** is a way for programs to talk to each other over the internet using standard web requests. Like a waiter taking your order (request) to the kitchen and bringing back food (response). This project exposes the simulation as an API so any AI agent can interact with it over the network.

---

## 4. How the Project Works — Step by Step

Here's the complete flow from start to finish:

```
STEP 1: Start a new episode
   → Send POST /reset with task="easy", "medium", or "hard"
   → Server initializes environment, returns first observation

STEP 2: AI receives observation (what's happening right now)
   → channel: what type of payment? (P2P/P2M/AutoPay)
   → risk_score: how suspicious is this transaction? (0-100)
   → kafka_lag: how backed up is the queue? (0-10000)
   → api_latency: how slow is the bank API? (0-5000ms)
   → rolling_p99: smoothed slowest 1% latency (0-5000ms)

STEP 3: AI makes 3 decisions (action)
   → risk_decision: 0=Approve, 1=Reject, 2=Challenge
   → infra_routing: 0=Normal, 1=Throttle, 2=CircuitBreaker
   → crypto_verify: 0=FullVerify, 1=SkipVerify

STEP 4: Send action to POST /step
   → Server applies decisions, simulates effects

STEP 5: Server calculates reward
   → Base: +0.8 for surviving the step
   → Bonuses: +0.05 for smart fraud handling
   → Penalties: -0.2 to -1.0 for bad decisions or SLA breaches

STEP 6: Check if episode is over
   → done=true if: 100 steps completed OR system crashed

STEP 7: Repeat from STEP 2 until done=true

STEP 8: Final score = average reward across all 100 steps
```

### Visual Flow Diagram

```
[AI Agent]
    |
    | POST /reset {"task": "medium"}
    v
[Spring Boot Server]
    |
    | Returns: {observation: {...}, done: false}
    v
[AI Agent decides]
    |
    | POST /step {"action": {"risk_decision": 1, "infra_routing": 0, "crypto_verify": 0}}
    v
[Server calculates reward]
    |
    | Returns: {observation: {...}, reward: 0.75, done: false, info: {...}}
    v
[AI Agent decides again] ← loops 100 times
    |
    v
[Episode ends] → Final score computed by Grader
```

---

## 5. Project Structure — What Each Folder Does

```
unified-fintech-risk-gateway/
│
├── unified_gateway.py       ← The main Python simulation (the "brain" of the environment)
├── graders.py               ← Scoring logic for Easy/Medium/Hard tasks
├── inference.py             ← The AI agent that calls the server and decides actions
├── openenv_bridge.py        ← Connects OpenEnv's testing tool to the Java server
├── openenv.yaml             ← Configuration file that tells OpenEnv what this env does
├── requirements.txt         ← List of Python packages needed
├── pyproject.toml           ← Python project settings
│
├── spring/                  ← The Java version of the project (currently running in production)
│   ├── pom.xml              ← Maven configuration (like requirements.txt but for Java)
│   └── src/
│       ├── main/java/com/ufrg/
│       │   ├── GatewayApplication.java    ← Start the Spring Boot server from here
│       │   ├── controller/
│       │   │   └── GatewayController.java ← Handles /reset, /step, /state HTTP requests
│       │   ├── env/
│       │   │   └── UnifiedFintechEnv.java ← Core simulation logic (Java version)
│       │   ├── grader/
│       │   │   ├── EasyGrader.java        ← Scores Easy task episodes
│       │   │   ├── MediumGrader.java      ← Scores Medium task episodes
│       │   │   ├── HardGrader.java        ← Scores Hard task episodes
│       │   │   └── GraderFactory.java     ← Picks the right grader based on task name
│       │   └── model/
│       │       ├── UFRGAction.java        ← Defines what an "action" looks like
│       │       ├── UFRGObservation.java   ← Defines what an "observation" looks like
│       │       └── UFRGReward.java        ← Defines what a "reward" looks like
│       └── test/java/com/ufrg/           ← Java tests
│
├── tests/                   ← Python tests
│   ├── test_foundation.py   ← Tests that models and reset/state work correctly
│   ├── test_step.py         ← Tests that rewards, crashes, and done logic work
│   └── test_graders.py      ← Tests that each task grader scores correctly
│
├── server/
│   └── app.py               ← Old FastAPI server (Python, legacy — kept for reference)
│
├── docs/                    ← Documentation
│   ├── BEGINNER_GUIDE.md    ← This file!
│   ├── MASTER_DOC.md        ← Deep technical reference
│   ├── PROJECT_REQUIREMENT.md ← Hackathon requirements
│   ├── COMPLIANCE_REPORT.md ← Checklist for OpenEnv compliance
│   └── JUDGE_READY_MANUAL.md ← For judges evaluating the submission
│
├── Dockerfile               ← Instructions to build a portable container
└── validate-submission.sh   ← Script to check everything is correct before submitting
```

---

## 6. The Three Task Levels

### Easy Task — "Normal Traffic Day"
- **What happens:** 100 regular UPI transactions, low fraud risk
- **Risk scores:** 5–30 (low)
- **Kafka lag:** Barely increases
- **Best strategy:** Approve everything, skip verification → collect 0.8/step
- **Score you need to pass:** 0.75+
- **Think of it as:** A quiet Tuesday morning with normal customers

### Medium Task — "Flash Sale (Flipkart Big Billion Day)"
- **What happens:** 80% normal + 20% massive volume spikes
- **Risk scores:** Very low (0–10) during spikes (legit customers rushing)
- **Kafka lag:** Surges +500 to +1000 per step during spikes — easily crashes!
- **Best strategy:** Throttle the server during spikes, allow traffic to drain safely
- **Score you need to pass:** 0.50+
- **Think of it as:** Black Friday — everyone's buying at once. Servers are struggling.

### Hard Task — "Botnet Storm"
- **What happens:** 100% of transactions are from bots/attackers (risk 85–100)
- **Kafka lag:** Steadily increases (+100 to +400 per step)
- **API latency:** Increases too (+50 to +150ms per step)
- **Best strategy:** Reject or challenge all payments while managing infrastructure
- **Score you need to pass:** 0.30+
- **Think of it as:** Your payment system is under a full-scale cyberattack

---

## 7. How Scoring (Rewards) Work

Every step (1 payment processed), the agent gets a reward between 0.0 and 1.0.

### Base Reward
- Just surviving a step without crashing: **+0.8**

### Bonuses (earning extra points for smart decisions)
| Situation | Smart Action | Bonus |
|---|---|---|
| High-risk payment (risk > 80) | Challenge it (action=2) | +0.05 |
| High-risk payment (risk > 80) | Full verification (crypto=0) | +0.03 |

### Penalties (losing points for bad decisions)
| Situation | Bad Action | Penalty |
|---|---|---|
| Normal traffic | Throttle unnecessarily | -0.2 |
| Flash sale spike | Throttle (blocks legit users) | -0.1 |
| P99 latency > 800ms | Any action | -0.3 |
| Circuit breaker activated | Any time | -0.5 |
| Approve + SkipVerify + HighRisk | All three at once | **-1.0 (crash!)** |
| Kafka lag > 4000 | — | **System crash → reward = 0.0** |

### What happens on a crash?
- Reward is set to **0.0** (worst possible)
- The episode ends immediately (`done = true`)
- The agent failed to protect the system

---

## 8. Running the Project — Prerequisites

Before running anything, you need to install some tools. Here's what's needed for each approach:

### Option A: Run Python version only
- **Python 3.10 or newer** — [Download from python.org](https://www.python.org/downloads/)
- **pip** (Python package installer — comes with Python)
- **Git** — [Download from git-scm.com](https://git-scm.com/downloads)

### Option B: Run Java/Spring Boot version
- **Java 21 (JDK)** — [Download from adoptium.net](https://adoptium.net/)
- **Maven 3.9+** — [Download from maven.apache.org](https://maven.apache.org/download.cgi)
- **Git**

### Option C: Run with Docker (easiest — recommended)
- **Docker Desktop** — [Download from docker.com](https://www.docker.com/products/docker-desktop/)
- **Git**

> **Tip for beginners:** Docker is the easiest option. It packages everything the project needs into a single "container" so you don't have to install Java, Maven, or Python separately.

---

## 9. Running on Windows

### Step 1: Clone the project
Open **Command Prompt** or **Git Bash** and run:
```bash
git clone https://github.com/your-username/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway
```

### Step 2A: Run Python version

**Install Python 3.10+** from [python.org](https://www.python.org/downloads/) — make sure to check "Add Python to PATH" during installation.

Open Command Prompt:
```bash
# Create a virtual environment (isolated Python space)
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should now see (venv) at the start of the prompt

# Install all required packages
pip install -e .

# Run Python tests to verify everything works
pytest tests/ -v
```

**Expected output:**
```
tests/test_foundation.py::test_reset_returns_valid_obs PASSED
tests/test_step.py::test_base_reward PASSED
tests/test_graders.py::test_easy_grader PASSED
...
All tests passed!
```

**Start the Python server:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open your browser and go to: `http://localhost:7860`

---

### Step 2B: Run Java/Spring Boot version

**Install Java 21:** Download JDK 21 from [adoptium.net](https://adoptium.net/)

During installation, select "Set JAVA_HOME variable" option.

**Install Maven:** Download from [maven.apache.org](https://maven.apache.org/download.cgi), extract it, and add `bin` folder to your PATH.

Verify installations:
```bash
java -version
# Should show: openjdk version "21.x.x"

mvn -version
# Should show: Apache Maven 3.x.x
```

Run the Spring Boot server:
```bash
cd spring
mvn spring-boot:run
```

**Expected output:**
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::  (v3.4.0)

Started GatewayApplication on port 7860
```

---

## 10. Running on macOS

### Step 1: Install Homebrew (macOS package manager)
Open **Terminal** and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Clone the project
```bash
git clone https://github.com/your-username/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway
```

### Step 3A: Run Python version
```bash
# Install Python 3.10+
brew install python@3.11

# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Step 3B: Run Java/Spring Boot version
```bash
# Install Java 21
brew install --cask temurin@21

# Install Maven
brew install maven

# Verify
java -version   # should show java 21
mvn -version    # should show Maven 3.x

# Run server
cd spring
mvn spring-boot:run
```

---

## 11. Running on Linux

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Git
sudo apt install -y git

# Clone project
git clone https://github.com/your-username/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway
```

**Python version:**
```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Java/Spring Boot version:**
```bash
# Install Java 21
sudo apt install -y openjdk-21-jdk

# Install Maven
sudo apt install -y maven

# Verify
java -version
mvn -version

# Run
cd spring
mvn spring-boot:run
```

### Fedora/RHEL/CentOS

```bash
sudo dnf install -y git java-21-openjdk-devel maven python3.11

# Same steps as above after this point
```

---

## 12. Running with Docker (Recommended for All Platforms)

Docker is the **simplest way** to run this project. It doesn't matter if you're on Windows, macOS, or Linux — Docker handles everything.

### Step 1: Install Docker Desktop
- Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
- Install and start it (you'll see the Docker whale icon in your system tray)

### Step 2: Clone the project
```bash
git clone https://github.com/your-username/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway
```

### Step 3: Build the Docker image
```bash
docker build -t ufrg .
```

This will:
1. Download a Maven + Java 21 base image
2. Compile the Java code
3. Package it into a small JRE-only image
4. Create a container called `ufrg`

**Expected output:**
```
[+] Building 45.2s
 => [build 1/5] FROM maven:3.9.6-eclipse-temurin-21
 => [build 2/5] COPY spring/pom.xml .
 => [build 3/5] RUN mvn dependency:go-offline
 => [build 4/5] COPY spring/src ./src
 => [build 5/5] RUN mvn package -DskipTests
 => [final 1/3] FROM eclipse-temurin:21-jre
 => [final 2/3] COPY --from=build /app/target/...jar app.jar
 => Successfully built image ufrg
```

### Step 4: Run the container
```bash
docker run -p 7860:7860 ufrg
```

The `-p 7860:7860` maps port 7860 inside the container to port 7860 on your machine.

Open your browser: `http://localhost:7860`

### Step 5: Verify it works
In a new terminal window:
```bash
# Health check
curl http://localhost:7860/

# Should return: {"status":"ok"} or similar
```

### Stopping the container
Press `Ctrl+C` in the terminal where Docker is running.

---

## 13. Testing the Project

### Python Tests
```bash
# Make sure virtual environment is activated first
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_foundation.py -v
pytest tests/test_step.py -v
pytest tests/test_graders.py -v
```

### Java Tests
```bash
cd spring
mvn test
```

### What the tests verify
| Test File | What it checks |
|---|---|
| `test_foundation.py` | Are the data models correct? Does reset() return valid data? |
| `test_step.py` | Are rewards calculated correctly? Does the system crash on lag > 4000? |
| `test_graders.py` | Does EasyGrader/MediumGrader/HardGrader score correctly? |
| `UnifiedFintechEnvFoundationTest.java` | Java equivalents of above |
| `UnifiedFintechEnvStepTest.java` | Java reward/crash tests |
| `TaskGraderTest.java` | Java grader tests |

### Run the validation script (checks everything at once)
```bash
chmod +x validate-submission.sh
./validate-submission.sh
```

This runs tests + Docker build + OpenEnv validation in sequence.

---

## 14. Using the Live API

The project is deployed live at:
```
https://unknown1321-unified-fintech-risk-gateway.hf.space
```

You can test it right now without installing anything using `curl` (command line) or any API tool like Postman.

### Start an Easy episode:
```bash
curl -X POST https://unknown1321-unified-fintech-risk-gateway.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

**Response:**
```json
{
  "observation": {
    "channel": 0.0,
    "risk_score": 12.5,
    "kafka_lag": 100.0,
    "api_latency": 95.0,
    "rolling_p99": 210.0
  },
  "done": false,
  "info": { "step": 0, "task": "easy" }
}
```

### Take a step (approve the payment, normal routing, skip verify):
```bash
curl -X POST https://unknown1321-unified-fintech-risk-gateway.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"risk_decision": 0, "infra_routing": 0, "crypto_verify": 1}}'
```

**Response:**
```json
{
  "observation": { "channel": 1.0, "risk_score": 18.0, ... },
  "reward": 0.8,
  "reward_breakdown": { "baseline": 0.8, "throttle_penalty": 0.0 },
  "done": false,
  "info": { "step": 1, "task": "easy", "crashed": false }
}
```

### Action value reference:
| Field | Value | Meaning |
|---|---|---|
| risk_decision | 0 | Approve the payment |
| risk_decision | 1 | Reject the payment |
| risk_decision | 2 | Challenge (ask for OTP, extra verification) |
| infra_routing | 0 | Normal — full speed |
| infra_routing | 1 | Throttle — slow down to reduce queue |
| infra_routing | 2 | Circuit Breaker — stop processing temporarily |
| crypto_verify | 0 | Full Verify — do complete cryptographic check |
| crypto_verify | 1 | Skip Verify — skip check for speed |

---

## 15. Common Errors and Fixes

### "Python not found" or "python is not recognized"
- Windows: Reinstall Python and check "Add Python to PATH"
- Or use `python3` instead of `python`

### "pip install fails" with permission error
```bash
pip install -e . --user
```

### "java.lang.UnsupportedClassVersionError"
Your Java version is too old. Run `java -version` — you need Java 21. Install JDK 21 from [adoptium.net](https://adoptium.net/).

### "mvn: command not found"
Maven is not installed or not in PATH. On macOS: `brew install maven`. On Linux: `sudo apt install maven`. On Windows: download and add to PATH manually.

### "Port 7860 is already in use"
Another process is using that port. Find and stop it:
```bash
# Windows
netstat -ano | findstr :7860
taskkill /PID <PID_NUMBER> /F

# macOS/Linux
lsof -ti:7860 | xargs kill -9
```

### "Docker daemon is not running"
Open Docker Desktop application and wait for it to fully start (the whale icon becomes steady).

### "Connection refused" when calling the API
The server hasn't started yet. Wait a few seconds after starting, then try again.

### Tests fail with "ModuleNotFoundError"
You forgot to activate the virtual environment:
```bash
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### Spring Boot starts but returns 404 on /reset
Make sure you're sending a POST request, not GET. The `/reset` endpoint requires a POST with a JSON body.

---

## 16. Glossary

| Term | Simple Definition |
|---|---|
| **UPI** | India's digital payment system (like Google Pay backend) |
| **SRE** | Site Reliability Engineer — person who keeps servers running |
| **Kafka** | A message queue system (conveyor belt for data) |
| **P99 Latency** | Speed of the slowest 1% of payments |
| **SLA** | Service Level Agreement — speed promise to customers |
| **RL / Reinforcement Learning** | AI learning by trial and error with rewards |
| **Gymnasium** | Python library for building RL environments |
| **OpenEnv** | Meta's framework for standardized AI agent environments |
| **LLM** | Large Language Model — AI like GPT-4, Claude, Qwen |
| **Episode** | One complete round of 100 transactions |
| **Observation** | What the AI can "see" about the current state |
| **Action** | The 3 decisions the AI makes each step |
| **Reward** | Score the AI earns for a step (0.0 to 1.0) |
| **REST API** | A way for programs to communicate over the internet |
| **Docker** | Tool that packages software into portable containers |
| **Spring Boot** | Java framework for building web servers quickly |
| **FastAPI** | Python framework for building web APIs quickly |
| **Maven** | Build tool for Java (manages packages and compilation) |
| **Virtual Environment** | Isolated Python installation for a specific project |
| **Grader** | Scoring logic that evaluates an AI agent's performance |
| **Circuit Breaker** | Safety mechanism that stops traffic to prevent crashes |
| **Throttle** | Slowing down request processing to reduce load |
| **Botnet** | Network of infected computers used for cyberattacks |
| **Fraud Risk Score** | A number (0–100) indicating how suspicious a payment is |
| **EMA** | Exponential Moving Average — a smoothed running average |
| **Multi-Objective** | AI that optimizes multiple goals at once |

---

## Quick Start Summary

```bash
# 1. Clone the project
git clone <repo-url>
cd unified-fintech-risk-gateway

# 2. Easiest way — Docker
docker build -t ufrg .
docker run -p 7860:7860 ufrg

# 3. Test the API
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# 4. That's it! You're running the Unified Fintech Risk Gateway.
```

---

*This document was written to make the project accessible to complete beginners. If something is unclear, please open an issue or ask a question in the project discussions.*
