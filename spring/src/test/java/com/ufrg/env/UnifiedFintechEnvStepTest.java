package com.ufrg.env;

import com.ufrg.model.UFRGAction;
import com.ufrg.model.UFRGObservation;
import com.ufrg.model.UFRGReward;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class UnifiedFintechEnvStepTest {

    private UnifiedFintechEnv env;

    @BeforeEach
    public void setUp() {
        env = new UnifiedFintechEnv();
    }

    private UFRGAction makeAction(int risk, int infra, int crypto) {
        return new UFRGAction(risk, infra, crypto);
    }
    
    private void setField(String fieldName, Object value) throws Exception {
        Field field = UnifiedFintechEnv.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(env, value);
    }

    // ── Return type contract ──────────────────────────────────────────────────────

    @Test
    public void testStepReturnsResult() {
        env.reset("easy");
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 0));
        assertNotNull(result);
        assertNotNull(result.observation);
        assertNotNull(result.reward);
        assertNotNull(result.done);
        assertNotNull(result.info);
    }

    // ── Reward clipping ──────────────────────────────────────────────────────────

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    public void testRewardValueInRange(String task) {
        env.reset(task);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 0));
        assertTrue(result.reward.value() >= 0.0 && result.reward.value() <= 1.0, 
                   "reward " + result.reward.value() + " out of [0,1]");
    }

    // ── Throttle penalty ──────────────────────────────────────────────────────────

    @Test
    public void testThrottlePenaltyNormalTraffic() throws Exception {
        env.reset("easy");
        setField("rollingLatency", 10.0);
        setField("rollingLag", 0.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result1 = env.step(makeAction(0, 0, 0));

        env.reset("easy");
        setField("rollingLatency", 10.0);
        setField("rollingLag", 0.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result2 = env.step(makeAction(0, 1, 0));

        double diff = result1.reward.value() - result2.reward.value();
        assertTrue(Math.abs(diff - 0.2) < 0.05, "expected ~0.2 diff, got " + diff);
    }

    // ── SLA breach penalty ────────────────────────────────────────────────────────

    @Test
    public void testSlaBreachPenalty() throws Exception {
        env.reset("easy");
        setField("rollingLag", 0.0);
        setField("currentObs", new UFRGObservation(0.0, 10.0, 0.0, 100.0, 2000.0));
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 0));
        
        double raw = (Double) result.info.get("reward_raw");
        assertTrue(Math.abs(raw - 0.5) < 0.01, "raw=" + raw);
    }

    // ── Circuit-breaker penalty ───────────────────────────────────────────────────

    @Test
    public void testCircuitBreakerPenalty() throws Exception {
        env.reset("easy");
        setField("rollingLatency", 10.0);
        setField("rollingLag", 0.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 2, 0));
        
        double raw = (Double) result.info.get("reward_raw");
        assertTrue(Math.abs(raw - 0.3) < 0.05, "raw=" + raw);
        assertTrue((Boolean) result.info.get("circuit_breaker_tripped"));
        
        double internalRollingLag = (Double) result.info.get("internal_rolling_lag");
        assertTrue(internalRollingLag < 50.0, "lag should be near 0 after CB");
    }

    // ── Catastrophic fraud ────────────────────────────────────────────────────────

    @Test
    public void testFraudGateClipsToZero() {
        env.reset("hard");
        // We know for hard task, risk > 85.0
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 1));
        assertEquals(0.0, result.reward.value(), 0.001);
        double obsRiskScore = (Double) result.info.get("obs_risk_score");
        assertTrue(obsRiskScore > 80.0);
    }

    // ── M2: Challenge bonus ───────────────────────────────────────────────────────

    @Test
    public void testChallengeBonusBeatsRejectOnHighRisk() throws Exception {
        env.reset("hard");
        setField("rollingLatency", 10.0);
        setField("rollingLag", 0.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> resultRej = env.step(makeAction(1, 0, 0));

        env.reset("hard");
        setField("rollingLatency", 10.0);
        setField("rollingLag", 0.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> resultChal = env.step(makeAction(2, 0, 0));

        assertTrue(resultChal.reward.value() > resultRej.reward.value(), 
            "challenge should beat reject");
    }

    // ── M2: Lag proximity warning ─────────────────────────────────────────────────

    @Test
    public void testLagProximityWarningInBreakdown() throws Exception {
        env.reset("easy");
        setField("rollingLag", 3500.0);
        setField("rollingLatency", 10.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 1));
        
        assertTrue(result.reward.breakdown().containsKey("lag_proximity_warning"), 
            "breakdown=" + result.reward.breakdown());
    }

    // ── Crash condition ───────────────────────────────────────────────────────────

    @Test
    public void testCrashForcesZeroRewardAndDone() throws Exception {
        env.reset("easy");
        setField("rollingLag", 4500.0);
        setField("rollingLatency", 10.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 0));
        
        assertEquals(0.0, result.reward.value(), 0.001);
        assertTrue(result.done);
    }

    // ── Circuit-breaker prevents crash ───────────────────────────────────────────

    @Test
    public void testCircuitBreakerPreventsCrash() throws Exception {
        env.reset("easy");
        setField("rollingLag", 4500.0);
        setField("rollingLatency", 10.0);
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 2, 0));
        
        Field currentStepField = UnifiedFintechEnv.class.getDeclaredField("currentStep");
        currentStepField.setAccessible(true);
        Field maxStepsField = UnifiedFintechEnv.class.getDeclaredField("maxSteps");
        maxStepsField.setAccessible(true);
        
        int currentStep = currentStepField.getInt(env);
        int maxSteps = maxStepsField.getInt(env);
        
        assertTrue(!result.done || currentStep >= maxSteps);
        
        double internalRollingLag = (Double) result.info.get("internal_rolling_lag");
        assertTrue(internalRollingLag < 50.0);
    }

    // ── max_steps triggers done ───────────────────────────────────────────────────

    @Test
    public void testMaxStepsTriggersDone() throws Exception {
        env.reset("easy");
        
        // mock maxSteps to 3 via reflection. Wait, maxSteps is final int 100 in UnifiedFintechEnv!
        // We can't safely change a primitive final field via pure Reflection in some modern Java versions 
        // without Unsafe, and it might get inlined by compiler.
        // Instead of overriding maxSteps, we can loop to 100 steps.
        Field maxStepsField = UnifiedFintechEnv.class.getDeclaredField("maxSteps");
        maxStepsField.setAccessible(true);
        int mSteps = maxStepsField.getInt(env);
        
        for (int i = 0; i < mSteps - 1; i++) {
            GymEnvironment.StepResult<UFRGObservation, UFRGReward> res = env.step(makeAction(0, 0, 0));
            // Reset lag so we don't naturally crash
            setField("rollingLag", 0.0);
            assertFalse(res.done, "done should be false at step " + (i + 1));
        }
        
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> finalRes = env.step(makeAction(0, 0, 0));
        assertTrue(finalRes.done, "done should be true at max steps");
    }

    // ── Info dict keys ────────────────────────────────────────────────────────────

    @Test
    public void testInfoContainsRequiredKeys() {
        env.reset("medium");
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(makeAction(0, 0, 0));
        
        Set<String> required = new HashSet<>(Arrays.asList(
            "step", "task", "event_type",
            "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
            "action_risk_decision", "action_infra_routing", "action_crypto_verify",
            "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
            "internal_rolling_lag", "internal_rolling_latency"
        ));
        
        assertTrue(result.info.keySet().containsAll(required), 
            "info keys missing. Expected subset: " + required + ". Found: " + result.info.keySet());
    }
}
