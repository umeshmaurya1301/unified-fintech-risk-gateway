package com.ufrg.inference;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ufrg.grader.GraderFactory;
import com.ufrg.grader.TaskGrader;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/**
 * InferenceRunner — Java equivalent of Python's {@code inference.py}.
 *
 * <p>Acts as a <em>decoupled HTTP client</em>. It never instantiates
 * {@code UnifiedFintechEnv} directly — all environment interaction goes
 * through the Spring Boot REST API:
 * <ul>
 *   <li>{@code POST /reset} → initialise a task, receive the first observation</li>
 *   <li>{@code POST /step}  → send an action, receive (obs, reward, done, info)</li>
 * </ul>
 *
 * <p>Only activated when the system property {@code inference.run=true} is set,
 * so normal server startup is unaffected.
 *
 * <h3>Environment variables</h3>
 * <ul>
 *   <li>{@code SPACE_URL}   — Base URL of the running server (default: http://localhost:7860)</li>
 *   <li>{@code DRY_RUN}     — "true" to use the heuristic fallback agent (no LLM needed)</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>
 *   # Standard dry-run (heuristic agent, no LLM calls):
 *   java -Dinference.run=true -DDRY_RUN=true -jar gateway.jar
 *
 *   # Against a remote space:
 *   java -Dinference.run=true -DSPACE_URL=https://your-space.hf.space -jar gateway.jar
 * </pre>
 */
@Component
public class InferenceRunner implements CommandLineRunner {

    private static final String DEFAULT_SPACE_URL = "http://localhost:7860";
    private static final String[] TASKS = {"easy", "medium", "hard"};

    private final ObjectMapper mapper = new ObjectMapper();

    @Override
    public void run(String... args) throws Exception {
        // Only activate when explicitly requested
        String runFlag = System.getProperty("inference.run",
                System.getenv().getOrDefault("INFERENCE_RUN", "false"));
        if (!"true".equalsIgnoreCase(runFlag.trim())) return;

        String spaceUrl = System.getenv().getOrDefault("SPACE_URL", DEFAULT_SPACE_URL)
                .stripTrailing().replaceAll("/$", "");
        boolean dryRun  = "true".equalsIgnoreCase(
                System.getenv().getOrDefault("DRY_RUN", "false").trim());

        System.out.printf("[INFERENCE] Starting. spaceUrl=%s dryRun=%b%n", spaceUrl, dryRun);

        HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .build();

        for (String task : TASKS) {
            runTask(http, spaceUrl, task, dryRun);
        }
    }

    // ── Per-task episode loop ─────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private void runTask(HttpClient http, String spaceUrl, String task, boolean dryRun) {
        List<Double>             stepRewards = new ArrayList<>();
        List<Map<String, Object>> trajectory = new ArrayList<>();
        int     currentStep = 0;
        boolean done        = false;
        double  taskScore   = 0.0;
        String  success     = "false";

        System.out.printf("[START] task=%s env=ufrg agent=%s%n",
                task, dryRun ? "heuristic" : "llm");

        try {
            // ── Reset ──────────────────────────────────────────────────────────
            Map<String, Object> obs = httpReset(http, spaceUrl, task);

            while (!done) {
                // ── Decide action ─────────────────────────────────────────────
                int[] action = heuristicAction(obs);   // always dry-run in Java port

                // ── Step ──────────────────────────────────────────────────────
                Map<String, Object> stepResp = httpStep(http, spaceUrl, action);

                obs        = (Map<String, Object>) stepResp.get("observation");
                double reward = asDouble(stepResp.get("reward"));
                done        = (Boolean) stepResp.getOrDefault("done", false);
                Map<String, Object> info = (Map<String, Object>) stepResp.getOrDefault("info", Map.of());

                stepRewards.add(reward);
                trajectory.add(info);
                currentStep++;

                String doneStr = done ? "true" : "false";
                System.out.printf("[STEP] step=%d action=%s reward=%.2f done=%s error=null%n",
                        currentStep, Arrays.toString(action), reward, doneStr);
            }

            // ── Grade ─────────────────────────────────────────────────────────
            TaskGrader grader = GraderFactory.getGrader(task);
            taskScore = grader.grade(trajectory);
            success   = taskScore >= 0.10 ? "true" : "false";

        } catch (Exception exc) {
            success   = "false";
            taskScore = 0.0;
            if (currentStep == 0) {
                System.out.printf("[STEP] step=1 action=null reward=0.00 done=true error=%s%n", exc.getMessage());
                stepRewards.add(0.0);
            }
        } finally {
            String rewardsCsv = stepRewards.isEmpty() ? "0.00" :
                    stepRewards.stream()
                               .map(r -> String.format("%.2f", r))
                               .reduce((a, b) -> a + "," + b)
                               .orElse("0.00");

            System.out.printf("[END] success=%s steps=%d score=%.2f rewards=%s%n",
                    success, Math.max(currentStep, stepRewards.size()), taskScore, rewardsCsv);
        }
    }

    // ── Heuristic agent (mirrors Python inference.py get_action dry_run=True) ─

    private int[] heuristicAction(Map<String, Object> obs) {
        double riskScore  = asDouble(obs.get("risk_score"));
        double kafkaLag   = asDouble(obs.get("kafka_lag"));
        double rollingP99 = asDouble(obs.get("rolling_p99"));

        int risk   = 0;   // Approve
        int infra  = 0;   // Normal
        int crypto = 1;   // SkipVerify (speed priority)

        if (riskScore > 80.0) {
            risk   = 1;   // Reject
            crypto = 0;   // FullVerify
        }
        if (kafkaLag > 3800.0) {
            infra = 2;    // CircuitBreaker
        } else if (kafkaLag > 3000.0) {
            infra = 1;    // Throttle
        }
        if (rollingP99 > 800.0 && infra == 0) {
            infra = 1;    // Throttle on SLA breach
        }

        return new int[]{risk, infra, crypto};
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Map<String, Object> httpReset(HttpClient http, String baseUrl, String task) throws Exception {
        String body = mapper.writeValueAsString(Map.of("task", task));
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/reset"))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200)
            throw new RuntimeException("POST /reset failed: HTTP " + resp.statusCode());
        Map<String, Object> data = mapper.readValue(resp.body(), Map.class);
        return (Map<String, Object>) data.get("observation");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> httpStep(HttpClient http, String baseUrl, int[] action) throws Exception {
        Map<String, Object> actionMap = Map.of(
                "risk_decision",  action[0],
                "infra_routing",  action[1],
                "crypto_verify",  action[2]
        );
        String body = mapper.writeValueAsString(Map.of("action", actionMap));
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/step"))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200)
            throw new RuntimeException("POST /step failed: HTTP " + resp.statusCode());
        return mapper.readValue(resp.body(), Map.class);
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private double asDouble(Object v) {
        if (v instanceof Number n) return n.doubleValue();
        return 0.0;
    }
}
