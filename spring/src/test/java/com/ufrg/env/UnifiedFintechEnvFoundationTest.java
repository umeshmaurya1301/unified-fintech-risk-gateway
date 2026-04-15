package com.ufrg.env;

import com.ufrg.model.UFRGAction;
import com.ufrg.model.UFRGObservation;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class UnifiedFintechEnvFoundationTest {

    private UnifiedFintechEnv env;
    private Validator validator;

    @BeforeEach
    public void setUp() {
        env = new UnifiedFintechEnv();
        validator = Validation.buildDefaultValidatorFactory().getValidator();
    }

    // ── UFRGAction validation ────────────────────────────────────────────────────

    @Test
    public void testActionValidConstruction() {
        UFRGAction a = new UFRGAction(1, 2, 0);
        Set<ConstraintViolation<UFRGAction>> violations = validator.validate(a);
        assertTrue(violations.isEmpty(), "Valid action should have no violations");
        assertEquals(1, a.riskDecision());
    }

    @Test
    public void testActionRejectsOutOfRange() {
        // risk_decision=3
        UFRGAction a1 = new UFRGAction(3, 0, 0);
        assertFalse(validator.validate(a1).isEmpty(), "riskDecision=3 should be invalid");

        // crypto_verify=2
        UFRGAction a2 = new UFRGAction(0, 0, 2);
        assertFalse(validator.validate(a2).isEmpty(), "cryptoVerify=2 should be invalid");

        // risk_decision=-1
        UFRGAction a3 = new UFRGAction(-1, 0, 0);
        assertFalse(validator.validate(a3).isEmpty(), "riskDecision=-1 should be invalid");
    }

    // ── reset() return contract ──────────────────────────────────────────────────

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    public void testResetReturnsFormat(String task) {
        GymEnvironment.ResetResult<UFRGObservation> result = env.reset(task);
        assertNotNull(result);
        assertNotNull(result.observation);
        assertNotNull(result.info);
    }

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    public void testResetInfoHasTaskKey(String task) {
        GymEnvironment.ResetResult<UFRGObservation> result = env.reset(task);
        assertEquals(task, result.info.get("task"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    public void testResetStoresCurrentTask(String task) throws Exception {
        env.reset(task);
        Field taskField = UnifiedFintechEnv.class.getDeclaredField("currentTask");
        taskField.setAccessible(true);
        assertEquals(task, taskField.get(env));
    }

    @ParameterizedTest
    @ValueSource(strings = {"easy", "medium", "hard"})
    public void testResetCurrentStepZero(String task) throws Exception {
        env.reset(task);
        Field stepField = UnifiedFintechEnv.class.getDeclaredField("currentStep");
        stepField.setAccessible(true);
        assertEquals(0, stepField.getInt(env));
    }

    @Test
    public void testResetDefaultsToEasyOnInvalid() throws Exception {
        env.reset("nightmare");
        Field taskField = UnifiedFintechEnv.class.getDeclaredField("currentTask");
        taskField.setAccessible(true);
        assertEquals("easy", taskField.get(env));
    }

    // ── state() ──────────────────────────────────────────────────────────────────

    @Test
    public void testStateReturnsObservation() {
        env.reset("easy");
        assertNotNull(env.state());
    }

    @Test
    public void testStateMatchesResetObs() {
        GymEnvironment.ResetResult<UFRGObservation> result = env.reset("easy");
        UFRGObservation st = env.state();
        assertEquals(result.observation.channel(), st.channel());
        assertEquals(result.observation.riskScore(), st.riskScore());
        assertEquals(result.observation.kafkaLag(), st.kafkaLag());
    }

    // ── _generate_transaction() per task ────────────────────────────────────────

    @Test
    public void testEasyRiskRange() throws Exception {
        env.reset("easy");
        Method generateTransaction = UnifiedFintechEnv.class.getDeclaredMethod("generateTransaction", String.class);
        generateTransaction.setAccessible(true);
        
        for (int i = 0; i < 50; i++) {
            UFRGObservation obs = (UFRGObservation) generateTransaction.invoke(env, "easy");
            assertTrue(obs.riskScore() >= 5.0 && obs.riskScore() <= 30.0, "Risk score out of range: " + obs.riskScore());
        }
    }

    @Test
    public void testHardRiskRange() throws Exception {
        env.reset("hard");
        Method generateTransaction = UnifiedFintechEnv.class.getDeclaredMethod("generateTransaction", String.class);
        generateTransaction.setAccessible(true);
        
        for (int i = 0; i < 50; i++) {
            UFRGObservation obs = (UFRGObservation) generateTransaction.invoke(env, "hard");
            assertTrue(obs.riskScore() >= 85.0, "Risk score should be >= 85 for hard task: " + obs.riskScore());
        }
    }

    @Test
    public void testMediumEventDistribution() throws Exception {
        env.reset("medium");
        Method generateTransaction = UnifiedFintechEnv.class.getDeclaredMethod("generateTransaction", String.class);
        generateTransaction.setAccessible(true);
        
        Field lastEventType = UnifiedFintechEnv.class.getDeclaredField("lastEventType");
        lastEventType.setAccessible(true);
        
        int normalCount = 0;
        int flashCount = 0;
        int numTrials = 200;
        
        for (int i = 0; i < numTrials; i++) {
            generateTransaction.invoke(env, "medium");
            String eventType = (String) lastEventType.get(env);
            if ("normal".equals(eventType)) normalCount++;
            else if ("flash_sale".equals(eventType)) flashCount++;
        }
        
        double normalPct = (normalCount / (double) numTrials) * 100;
        double flashPct = (flashCount / (double) numTrials) * 100;
        
        assertTrue(normalPct > 55 && normalPct < 95, "normal_pct out of bounds: " + normalPct);
        assertTrue(flashPct > 5 && flashPct < 45, "flash_pct out of bounds: " + flashPct);
    }
}
