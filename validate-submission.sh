#!/usr/bin/env bash
# =============================================================================
# validate-submission.sh — Pre-Submission Validation Script (Spring Boot edition)
# =============================================================================
# Run this script before submitting to the Meta OpenEnv Hackathon.
# It checks all requirements from the Pre-Submission Checklist:
#   1. Maven test suite passes (Java JUnit 5 tests)
#   2. HF Space is live and POST /reset returns 200
#   3. docker build succeeds locally (multi-stage Java build)
#   4. openenv validate passes (via Python bridge)
#   5. Bridge self-test passes (openenv_bridge.py)
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh
#
# Override the Space URL:
#   SPACE_URL=https://your-space.hf.space ./validate-submission.sh
# =============================================================================

set -e   # abort on first error

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'   # No Colour

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; EXIT_CODE=1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

EXIT_CODE=0

echo "============================================================"
echo "  UFRG Pre-Submission Validation (Spring Boot)"
echo "============================================================"
echo ""

# ── Config ───────────────────────────────────────────────────────────────────
SPACE_URL="${SPACE_URL:-https://unknown1321-unified-fintech-risk-gateway.hf.space}"
IMAGE_NAME="ufrg-validate"

# ── Check 1: Maven unit tests ─────────────────────────────────────────────────
echo "── Check 1: Maven / JUnit 5 test suite ────────────────────"
if ! command -v mvn &> /dev/null; then
    info "mvn not found in PATH — skipping Java unit tests"
else
    info "Running mvn test in ./spring ..."
    if mvn test -f spring/pom.xml -q 2>&1; then
        pass "All JUnit 5 tests passed (mvn test)"
    else
        fail "mvn test FAILED — review spring/src/test for errors"
    fi
fi

echo ""

# ── Check 2: HF Space health ─────────────────────────────────────────────────
echo "── Check 2: HF Space liveness ──────────────────────────────"
info "Pinging GET ${SPACE_URL} ..."

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${SPACE_URL}" || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
    pass "GET / returned HTTP 200"
else
    fail "GET / returned HTTP ${HTTP_STATUS} (Space may be sleeping or URL is wrong)"
fi

info "Checking POST ${SPACE_URL}/reset for task=easy ..."
RESET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 \
    -X POST "${SPACE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d '{"task": "easy"}' || echo "000")

if [ "$RESET_STATUS" = "200" ]; then
    pass "POST /reset returned HTTP 200"
else
    fail "POST /reset returned HTTP ${RESET_STATUS}"
fi

echo ""

# ── Check 3: Docker build ─────────────────────────────────────────────────────
echo "── Check 3: Docker build (multi-stage Java) ────────────────"
if ! command -v docker &> /dev/null; then
    info "Docker not found — skipping build check"
else
    info "Building Docker image '${IMAGE_NAME}' (multi-stage: Maven → JRE 21) ..."
    if docker build -t "${IMAGE_NAME}" . --quiet; then
        pass "docker build succeeded"
        # Cleanup
        docker rmi "${IMAGE_NAME}" --force > /dev/null 2>&1 || true
    else
        fail "docker build FAILED — review Dockerfile"
    fi
fi

echo ""

# ── Check 4: openenv validate ─────────────────────────────────────────────────
echo "── Check 4: openenv validate ────────────────────────────────"
if ! command -v openenv &> /dev/null; then
    info "openenv CLI not found — installing openenv-core ..."
    pip3 install openenv-core --quiet 2>/dev/null || pip install openenv-core --quiet
fi

# openenv validate reads openenv.yaml and imports the bridge entry_point
if openenv validate .; then
    pass "openenv validate PASSED"
else
    fail "openenv validate FAILED — review openenv.yaml and openenv_bridge.py"
fi

echo ""

# ── Check 5: Bridge self-test ─────────────────────────────────────────────────
echo "── Check 5: openenv_bridge.py self-test ────────────────────"
info "Running bridge self-test against ${SPACE_URL} ..."
if SPACE_URL="${SPACE_URL}" python3 openenv_bridge.py 2>&1; then
    pass "openenv_bridge.py self-test PASSED"
else
    fail "openenv_bridge.py self-test FAILED — ensure server is running at ${SPACE_URL}"
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN}  ✅ ALL CHECKS PASSED — Safe to submit!${NC}"
else
    echo -e "${RED}  ❌ SOME CHECKS FAILED — Fix issues before submitting.${NC}"
fi
echo "============================================================"

exit $EXIT_CODE
