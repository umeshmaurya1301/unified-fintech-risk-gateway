package com.ufrg.grader;

import java.util.List;
import java.util.Map;

/**
 * MediumGrader — "Flash-Sale Uptime Ratio".
 *
 * <p>Scores the <em>uptime ratio</em> — fraction of steps completed without a
 * system crash OR an SLA breach, plus a bonus for intelligent throttling during
 * flash-sale spikes.
 *
 * <h3>Scoring formula</h3>
 * <p>A step is "clean" when:
 * <ol>
 *   <li>{@code crashed == false}</li>
 *   <li>{@code obs_rolling_p99 ≤ 800 ms}</li>
 * </ol>
 * Bonus: +0.1 per correctly-throttled flash_sale step (normalised over total steps, max 0.1).
 * Final score = clean_steps / total_steps + normalised_bonus, clipped to [0.01, 0.99].
 */
public class MediumGrader implements TaskGrader {

    private static final double SLA_THRESHOLD_MS = 800.0;
    private static final double THROTTLE_BONUS   = 0.1;

    @Override
    public double grade(List<Map<String, Object>> trajectory) {
        if (trajectory == null || trajectory.isEmpty()) return FLOOR;

        int    cleanSteps    = 0;
        double throttleBonus = 0.0;
        int    total         = trajectory.size();

        for (Map<String, Object> step : trajectory) {
            boolean crashed   = getBool(step, "crashed", false);
            double  p99       = getDouble(step, "obs_rolling_p99", 0.0);
            String  eventType = getString(step, "event_type", "normal");
            int     infra     = getInt(step, "action_infra_routing", 0);

            if (!crashed && p99 <= SLA_THRESHOLD_MS) {
                cleanSteps++;
            }
            if ("flash_sale".equals(eventType) && infra == 1) {
                throttleBonus += THROTTLE_BONUS;
            }
        }

        double baseScore       = (double) cleanSteps / total;
        double normalisedBonus = Math.min(throttleBonus / total, 0.1);
        return clip(baseScore + normalisedBonus);
    }
}
