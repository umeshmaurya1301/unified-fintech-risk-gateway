package com.ufrg.grader;

import java.util.List;
import java.util.Map;

/**
 * EasyGrader — "Normal Traffic Baseline".
 *
 * <p>Scores the fraction of steps where the agent achieved the <em>baseline
 * reward</em> (≥ 0.8) without unnecessary throttling or circuit-breaker use.
 *
 * <h3>Scoring formula</h3>
 * <ul>
 *   <li>Full credit (+1.0) if reward_final ≥ 0.8 AND infra_routing == 0</li>
 *   <li>Partial credit (+0.5) if reward_final ≥ 0.8 but throttle/CB used</li>
 *   <li>No credit (+0.0) otherwise</li>
 * </ul>
 * Final score = total_credit / total_steps, clipped to [0.01, 0.99].
 */
public class EasyGrader implements TaskGrader {

    @Override
    public double grade(List<Map<String, Object>> trajectory) {
        if (trajectory == null || trajectory.isEmpty()) return FLOOR;

        double totalCredit = 0.0;

        for (Map<String, Object> step : trajectory) {
            double reward = getDouble(step, "reward_final", 0.0);
            int    infra  = getInt(step, "action_infra_routing", 0);

            if (reward >= 0.8) {
                totalCredit += (infra == 0) ? 1.0 : 0.5;  // Normal=full, Throttle/CB=partial
            }
        }

        double rawScore = totalCredit / trajectory.size();
        return clip(rawScore);
    }
}
