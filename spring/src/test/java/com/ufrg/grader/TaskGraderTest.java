package com.ufrg.grader;


import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * TaskGraderTest — JUnit 5 translation of Python's tests/test_graders.py.
 *
 * Boundary contract:
 *  - All graders return float in [0.01, 0.99] (inclusive).
 *  - Empty trajectory always scores 0.01 (floor sentinel).
 *  - Perfect-agent trajectories score 0.99 (ceiling sentinel, NOT 1.0).
 *  - All-failure trajectories score 0.01 (floor sentinel, NOT 0.0).
 */
class TaskGraderTest {

    // ── Shared fixture builder ────────────────────────────────────────────────

    private static Map<String, Object> makeStep(
            double rewardFinal,
            int    infra,
            boolean crashed,
            double p99,
            String eventType,
            double riskScore,
            int    decision,
            int    crypto) {
        Map<String, Object> step = new HashMap<>();
        step.put("reward_final",          rewardFinal);
        step.put("action_infra_routing",  infra);
        step.put("crashed",               crashed);
        step.put("obs_rolling_p99",       p99);
        step.put("event_type",            eventType);
        step.put("obs_risk_score",        riskScore);
        step.put("action_risk_decision",  decision);
        step.put("action_crypto_verify",  crypto);
        return step;
    }



    private static List<Map<String, Object>> repeat(Map<String, Object> step, int n) {
        List<Map<String, Object>> traj = new ArrayList<>();
        for (int i = 0; i < n; i++) traj.add(new HashMap<>(step));
        return traj;
    }

    // ── GraderFactory ─────────────────────────────────────────────────────────

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    void testGetGraderReturnsCorrectType(String task) {
        TaskGrader g = GraderFactory.getGrader(task);
        assertNotNull(g);
        assertTrue(switch (task) {
            case "easy"   -> g instanceof EasyGrader;
            case "medium" -> g instanceof MediumGrader;
            case "hard"   -> g instanceof HardGrader;
            default       -> false;
        });
    }

    @Test
    void testGetGraderInvalidTaskThrows() {
        assertThrows(IllegalArgumentException.class, () -> GraderFactory.getGrader("legendary"));
    }

    // =========================================================================
    // EasyGrader
    // =========================================================================

    @Test
    void easyEmptyTrajectoryScoresFloor() {
        assertEquals(0.01, new EasyGrader().grade(Collections.emptyList()), 0.001);
    }

    @Test
    void easyPerfectAgentScoresCeiling() {
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 0, false, 50.0, "normal", 20.0, 0, 0), 20);
        assertEquals(0.99, new EasyGrader().grade(traj), 0.001);
    }

    @Test
    void easyThrottleHeavyScoresHalf() {
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 1, false, 50.0, "normal", 20.0, 0, 0), 10);
        assertEquals(0.5, new EasyGrader().grade(traj), 0.001);
    }

    @Test
    void easyLowRewardScoresFloor() {
        List<Map<String, Object>> traj = repeat(makeStep(0.3, 0, false, 50.0, "normal", 20.0, 0, 0), 10);
        assertEquals(0.01, new EasyGrader().grade(traj), 0.001);
    }

    @Test
    void easyScoreInRange() {
        List<Map<String, Object>> traj = new ArrayList<>();
        traj.addAll(repeat(makeStep(0.8, 0, false, 50.0, "normal", 20.0, 0, 0), 5));
        traj.addAll(repeat(makeStep(0.0, 2, false, 50.0, "normal", 20.0, 0, 0), 5));
        double score = new EasyGrader().grade(traj);
        assertTrue(score >= 0.01 && score <= 0.99, "score=" + score);
    }

    @Test
    void easyDeterministic() {
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 0, false, 50.0, "normal", 20.0, 0, 0), 10);
        EasyGrader g = new EasyGrader();
        assertEquals(g.grade(traj), g.grade(traj));
    }

    @Test
    void easyPartialVsFullCredit() {
        List<Map<String, Object>> full    = repeat(makeStep(0.8, 0, false, 50.0, "normal", 20.0, 0, 0), 10);
        List<Map<String, Object>> partial = repeat(makeStep(0.8, 1, false, 50.0, "normal", 20.0, 0, 0), 10);
        assertTrue(new EasyGrader().grade(full) > new EasyGrader().grade(partial));
    }

    @Test
    void easySingleStepPerfect() {
        assertEquals(0.99, new EasyGrader().grade(List.of(makeStep(0.8, 0, false, 50.0, "normal", 20.0, 0, 0))), 0.001);
    }

    @Test
    void easySingleStepFailure() {
        assertEquals(0.01, new EasyGrader().grade(List.of(makeStep(0.0, 0, false, 50.0, "normal", 20.0, 0, 0))), 0.001);
    }

    // =========================================================================
    // MediumGrader
    // =========================================================================

    @Test
    void mediumEmptyTrajectoryScoresFloor() {
        assertEquals(0.01, new MediumGrader().grade(Collections.emptyList()), 0.001);
    }

    @Test
    void mediumAllCrashesScoresFloor() {
        List<Map<String, Object>> traj = repeat(makeStep(0.0, 0, true, 900.0, "normal", 20.0, 0, 0), 10);
        assertEquals(0.01, new MediumGrader().grade(traj), 0.001);
    }

    @Test
    void mediumCleanStepsScoreNearCeiling() {
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 0, false, 400.0, "normal", 20.0, 0, 0), 20);
        assertEquals(0.99, new MediumGrader().grade(traj), 0.001);
    }

    @Test
    void mediumFlashSaleThrottleBonus() {
        List<Map<String, Object>> base      = repeat(makeStep(0.8, 0, false, 400.0, "normal",     20.0, 0, 0), 10);
        List<Map<String, Object>> withBonus = new ArrayList<>(base);
        withBonus.addAll(repeat(makeStep(0.8, 1, false, 400.0, "flash_sale", 20.0, 0, 0), 2));
        List<Map<String, Object>> noBonus = new ArrayList<>(base);
        noBonus.addAll(repeat(makeStep(0.8, 0, false, 400.0, "flash_sale",  20.0, 0, 0), 2));
        assertTrue(new MediumGrader().grade(withBonus) >= new MediumGrader().grade(noBonus));
    }

    @Test
    void mediumScoreInRange() {
        List<Map<String, Object>> traj = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            traj.add(makeStep(0.8, 0, (i % 3 == 0), 200.0, "normal", 20.0, 0, 0));
        }
        double score = new MediumGrader().grade(traj);
        assertTrue(score >= 0.01 && score <= 0.99, "score=" + score);
    }

    @Test
    void mediumDeterministic() {
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 0, false, 250.0, "normal", 20.0, 0, 0), 10);
        MediumGrader g = new MediumGrader();
        assertEquals(g.grade(traj), g.grade(traj));
    }

    @Test
    void mediumSlaBreachReducesCleanSteps() {
        List<Map<String, Object>> cleanOnly  = repeat(makeStep(0.8, 0, false, 400.0, "normal", 20.0, 0, 0), 10);
        List<Map<String, Object>> withBreach = new ArrayList<>();
        withBreach.addAll(repeat(makeStep(0.8, 0, false, 400.0, "normal", 20.0, 0, 0), 5));
        withBreach.addAll(repeat(makeStep(0.8, 0, false, 900.0, "normal", 20.0, 0, 0), 5));
        assertTrue(new MediumGrader().grade(cleanOnly) > new MediumGrader().grade(withBreach));
    }

    @Test
    void mediumNormalisedBonusDoesNotExceed01() {
        // All flash_sale + throttle → bonus capped; final raw capped to 0.99
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 1, false, 400.0, "flash_sale", 20.0, 0, 0), 20);
        assertEquals(0.99, new MediumGrader().grade(traj), 0.001);
    }

    // =========================================================================
    // HardGrader
    // =========================================================================

    @Test
    void hardEmptyTrajectoryScoresFloor() {
        assertEquals(0.01, new HardGrader().grade(Collections.emptyList()), 0.001);
    }

    @Test
    void hardPerfectRejectFullVerifyNocrashScoresCeiling() {
        // FCR=1.0+0.1 bonus capped=1.0, crash_penalty=0, sla_bonus=0.1 → raw=1.10 → 0.99
        List<Map<String, Object>> traj = repeat(makeStep(1.0, 0, false, 400.0, "botnet_attack", 95.0, 1, 0), 10);
        assertEquals(0.99, new HardGrader().grade(traj), 0.001);
    }

    @Test
    void hardAllApproveScoresFloor() {
        // All approve on high-risk + p99>800 → FCR=0, sla_bonus=0 → floor
        List<Map<String, Object>> traj = repeat(makeStep(0.0, 0, false, 1000.0, "botnet_attack", 95.0, 0, 0), 10);
        assertEquals(0.01, new HardGrader().grade(traj), 0.001);
    }

    @Test
    void hardChallengeScoresLessThanReject() {
        List<Map<String, Object>> rejectTraj    = repeat(makeStep(1.0, 0, false, 1000.0, "botnet_attack", 95.0, 1, 0), 10);
        List<Map<String, Object>> challengeTraj = repeat(makeStep(1.0, 0, false, 1000.0, "botnet_attack", 95.0, 2, 0), 10);
        assertTrue(new HardGrader().grade(rejectTraj) > new HardGrader().grade(challengeTraj));
    }

    @Test
    void hardCrashPenalisesScore() {
        List<Map<String, Object>> noCrash   = repeat(makeStep(1.0, 0, false, 400.0, "botnet_attack", 95.0, 1, 0), 10);
        List<Map<String, Object>> withCrash = repeat(makeStep(1.0, 0, true,  400.0, "botnet_attack", 95.0, 1, 0), 10);
        assertTrue(new HardGrader().grade(noCrash) > new HardGrader().grade(withCrash));
    }

    @Test
    void hardFullVerifyBonusApplied() {
        List<Map<String, Object>> mixed = new ArrayList<>();
        mixed.addAll(repeat(makeStep(1.0, 0, false, 400.0, "botnet_attack", 95.0, 1, 0), 5)); // catch
        mixed.addAll(repeat(makeStep(0.0, 0, false, 400.0, "botnet_attack", 95.0, 0, 0), 5)); // miss
        List<Map<String, Object>> fullVerify = new ArrayList<>();
        List<Map<String, Object>> skipVerify = new ArrayList<>();
        for (Map<String, Object> s : mixed) {
            Map<String, Object> fv = new HashMap<>(s); fv.put("action_crypto_verify", 0); fullVerify.add(fv);
            Map<String, Object> sv = new HashMap<>(s); sv.put("action_crypto_verify", 1); skipVerify.add(sv);
        }
        assertTrue(new HardGrader().grade(fullVerify) >= new HardGrader().grade(skipVerify));
    }

    @Test
    void hardScoreInRange() {
        List<Map<String, Object>> traj = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            traj.add(makeStep(0.8, 0, (i % 4 == 0), 500.0, "botnet_attack", 90.0, (i % 2), 0));
        }
        double score = new HardGrader().grade(traj);
        assertTrue(score >= 0.01 && score <= 0.99, "score=" + score);
    }

    @Test
    void hardDeterministic() {
        List<Map<String, Object>> traj = repeat(makeStep(1.0, 0, false, 400.0, "botnet_attack", 92.0, 1, 0), 10);
        HardGrader g = new HardGrader();
        assertEquals(g.grade(traj), g.grade(traj));
    }

    @Test
    void hardNoHighRiskStepsGivesFloor() {
        // No high-risk steps → FCR=0.0, sla_bonus=0.1 → raw=0.10 → valid fraction
        List<Map<String, Object>> traj = repeat(makeStep(0.8, 0, false, 200.0, "normal", 30.0, 0, 0), 10);
        double score = new HardGrader().grade(traj);
        assertTrue(score >= 0.01 && score <= 0.99, "score=" + score);
    }

    @Test
    void hardHeavyCrashPenaltyFlooredAt001() {
        // 10 crashed steps → penalty=1.5; even perfect FCR gives negative → floor
        List<Map<String, Object>> traj = repeat(makeStep(1.0, 0, true, 400.0, "botnet_attack", 95.0, 1, 0), 10);
        assertEquals(0.01, new HardGrader().grade(traj), 0.001);
    }

    @Test
    void hardSlasBonusImprovesScore() {
        List<Map<String, Object>> mixedBase = new ArrayList<>();
        mixedBase.addAll(repeat(makeStep(1.0, 0, false, 400.0, "botnet_attack", 95.0, 1, 0), 5));
        mixedBase.addAll(repeat(makeStep(0.0, 0, false, 400.0, "botnet_attack", 95.0, 0, 1), 5));
        List<Map<String, Object>> slaOk   = new ArrayList<>();
        List<Map<String, Object>> slaFail = new ArrayList<>();
        for (Map<String, Object> s : mixedBase) {
            Map<String, Object> ok   = new HashMap<>(s); ok.put("obs_rolling_p99",   400.0); slaOk.add(ok);
            Map<String, Object> fail = new HashMap<>(s); fail.put("obs_rolling_p99", 1000.0); slaFail.add(fail);
        }
        assertTrue(new HardGrader().grade(slaOk) > new HardGrader().grade(slaFail));
    }

    // =========================================================================
    // Cross-grader boundary contract
    // =========================================================================

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    void allGradersFloorAt001(String task) {
        TaskGrader grader = GraderFactory.getGrader(task);
        List<Map<String, Object>> worst = repeat(makeStep(0.0, 2, true, 5000.0, "normal", 95.0, 0, 1), 10);
        assertTrue(grader.grade(worst) >= 0.01, "floor violated for task=" + task);
    }

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    void allGradersCeilingAt099(String task) {
        TaskGrader grader = GraderFactory.getGrader(task);
        List<Map<String, Object>> best = repeat(makeStep(1.0, 0, false, 10.0, "normal", 95.0, 1, 0), 100);
        assertTrue(grader.grade(best) <= 0.99, "ceiling violated for task=" + task);
    }

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    void emptyAlwaysScoresFloor(String task) {
        assertEquals(0.01, GraderFactory.getGrader(task).grade(Collections.emptyList()), 0.001);
    }
}
