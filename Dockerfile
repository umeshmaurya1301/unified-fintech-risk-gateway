# ─────────────────────────────────────────────────────────────────────────────
# Unified Fintech Risk Gateway (UFRG) — Spring Boot Container
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the original Python FastAPI image with a Java 21 + Spring Boot build.
#
# Two usage modes:
#
#   1. API server (default — used by HF Spaces and openenv validate):
#        docker run -p 7860:7860 ufrg
#
#   2. Inference / baseline scoring (heuristic dry-run agent):
#        docker run --rm \
#          -e SPACE_URL=http://localhost:7860 \
#          -e INFERENCE_RUN=true \
#          -e DRY_RUN=true \
#          ufrg java -Dinference.run=true -jar /app/gateway.jar
#
#   3. OpenEnv validation bridge (Python compatibility shim):
#        docker run --rm ufrg python /app/openenv_bridge.py
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build the Spring Boot fat JAR ────────────────────────────────────
FROM maven:3.9.6-eclipse-temurin-21 AS builder

WORKDIR /build

# Copy pom.xml first for dependency caching
COPY spring/pom.xml ./pom.xml
RUN mvn dependency:go-offline -q

# Copy source and build
COPY spring/src ./src
RUN mvn package -DskipTests -q

# ── Stage 2: Minimal runtime image ───────────────────────────────────────────
FROM eclipse-temurin:21-jre-jammy

# Metadata
LABEL maintainer="Umesh Maurya <umeshmaurya1301>" \
      description="Unified Fintech Risk Gateway — Spring Boot RL Environment" \
      version="2.0.0"

# Install Python 3 (for openenv validate CLI bridge)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fat JAR
COPY --from=builder /build/target/unified-fintech-risk-gateway-*.jar /app/gateway.jar

# Copy openenv bridge script and its requirements
COPY openenv_bridge.py /app/openenv_bridge.py
COPY openenv.yaml      /app/openenv.yaml

# Install only the minimal Python deps needed for openenv CLI bridge
RUN pip3 install --no-cache-dir openenv-core 2>/dev/null || true

# ── Port configuration ────────────────────────────────────────────────────────
# HuggingFace Spaces routes external traffic to port 7860
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Default: run the Spring Boot server ──────────────────────────────────────
CMD ["java", "-jar", "/app/gateway.jar"]