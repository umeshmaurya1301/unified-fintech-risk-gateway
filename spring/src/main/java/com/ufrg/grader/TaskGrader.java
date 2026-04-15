package com.ufrg.grader;

import java.util.List;
import java.util.Map;

/**
 * TaskGrader — per-task programmatic grader interface.
 *
 * <p>Each grader receives a <em>trajectory</em> — a {@code List<Map<String,Object>>}
 * where every element is the {@code info} dictionary returned by one call to
 * {@code env.step()}.  The info map schema mirrors the Python implementation:
 *
 * <pre>
 *   step                    int
 *   task                    String  — "easy" | "medium" | "hard"
 *   event_type              String  — "normal" | "flash_sale" | "botnet_attack"
 *   obs_risk_score          double
 *   obs_kafka_lag           double
 *   obs_rolling_p99         double
 *   action_risk_decision    int     — 0=Approve 1=Reject 2=Challenge
 *   action_infra_routing    int     — 0=Normal  1=Throttle 2=CircuitBreaker
 *   action_crypto_verify    int     — 0=FullVerify 1=SkipVerify
 *   reward_final            double  — clipped reward [0.0, 1.0]
 *   circuit_breaker_tripped boolean
 *   crashed                 boolean
 *   done                    boolean
 * </pre>
 *
 * <p>All implementations guarantee a return value in <strong>[0.01, 0.99]</strong>.
 */
public interface TaskGrader {

    /**
     * Score an entire episode trajectory.
     *
     * @param trajectory list of step info-maps from consecutive {@code env.step()} calls
     * @return score in [0.01, 0.99]
     */
    double grade(List<Map<String, Object>> trajectory);

    // ── Boundary helpers (shared by all graders) ──────────────────────────────

    double FLOOR   = 0.01;
    double CEILING = 0.99;

    /** Applies floor/ceiling and rounds to 2 decimal places. */
    default double clip(double raw) {
        double clamped = Math.max(FLOOR, Math.min(CEILING, raw));
        return Math.round(clamped * 100.0) / 100.0;
    }

    // ── Typed safe accessors ──────────────────────────────────────────────────

    default double getDouble(Map<String, Object> step, String key, double defaultValue) {
        Object v = step.get(key);
        if (v instanceof Number n) return n.doubleValue();
        return defaultValue;
    }

    default int getInt(Map<String, Object> step, String key, int defaultValue) {
        Object v = step.get(key);
        if (v instanceof Number n) return n.intValue();
        return defaultValue;
    }

    default boolean getBool(Map<String, Object> step, String key, boolean defaultValue) {
        Object v = step.get(key);
        if (v instanceof Boolean b) return b;
        return defaultValue;
    }

    default String getString(Map<String, Object> step, String key, String defaultValue) {
        Object v = step.get(key);
        if (v instanceof String s) return s;
        return defaultValue;
    }
}
