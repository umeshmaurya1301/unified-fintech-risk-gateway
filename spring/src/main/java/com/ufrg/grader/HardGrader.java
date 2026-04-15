package com.ufrg.grader;

import java.util.List;
import java.util.Map;

/**
 * HardGrader — "Botnet Fraud-Catch Rate".
 *
 * <p>Scores the <em>fraud-catch rate</em> minus a heavy penalty for any crash,
 * plus a small SLA bonus.
 *
 * <h3>Scoring formula</h3>
 * <h4>Fraud Catch Rate (FCR)</h4>
 * For each step where {@code obs_risk_score > 80}:
 * <ul>
 *   <li>Reject (1) → +1.0</li>
 *   <li>Challenge (2) → +0.8</li>
 *   <li>Approve (0) → +0.0 (miss)</li>
 *   <li>Bonus: if correctly caught AND crypto_verify == 0 → +0.1 (capped at 1.0)</li>
 * </ul>
 * FCR = total_catch_score / total_high_risk_steps
 *
 * <h4>Crash penalty</h4>
 * Each crashed step deducts −0.15 from the final score (post-FCR).
 *
 * <h4>SLA bonus</h4>
 * 0.1 × (sla_ok_steps / total) added after crash penalty.
 *
 * Final = max(0.0, FCR − crash_penalty + sla_bonus), clipped to [0.01, 0.99].
 */
public class HardGrader implements TaskGrader {

    private static final double RISK_THRESHOLD = 80.0;
    private static final double CRASH_PENALTY  = 0.15;

    @Override
    public double grade(List<Map<String, Object>> trajectory) {
        if (trajectory == null || trajectory.isEmpty()) return FLOOR;

        int    highRiskSteps   = 0;
        double totalCatchScore = 0.0;
        int    crashCount      = 0;
        int    slaOkSteps      = 0;
        int    total           = trajectory.size();

        for (Map<String, Object> step : trajectory) {
            double  riskScore = getDouble(step, "obs_risk_score", 0.0);
            int     decision  = getInt(step, "action_risk_decision", 0);
            int     crypto    = getInt(step, "action_crypto_verify", 0);
            boolean crashed   = getBool(step, "crashed", false);
            double  p99       = getDouble(step, "obs_rolling_p99", 0.0);

            if (crashed) crashCount++;
            if (p99 <= 800.0) slaOkSteps++;

            if (riskScore > RISK_THRESHOLD) {
                highRiskSteps++;

                double catchScore = switch (decision) {
                    case 1  -> 1.0;   // Reject
                    case 2  -> 0.8;   // Challenge
                    default -> 0.0;   // Approve = miss
                };

                // FullVerify bonus on a correctly caught threat
                if (catchScore > 0 && crypto == 0) {
                    catchScore = Math.min(1.0, catchScore + 0.1);
                }
                totalCatchScore += catchScore;
            }
        }

        double fcr          = (highRiskSteps > 0) ? totalCatchScore / highRiskSteps : 0.0;
        double crashPenalty = crashCount * CRASH_PENALTY;
        double slaBonus     = 0.1 * ((double) slaOkSteps / total);

        return clip(fcr - crashPenalty + slaBonus);
    }
}
