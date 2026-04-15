package com.ufrg.env;

import com.ufrg.model.UFRGAction;
import com.ufrg.model.UFRGObservation;
import com.ufrg.model.UFRGReward;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class UnifiedFintechEnvTest {

    @Test
    public void testResetInitializesStateCorrectly() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        GymEnvironment.ResetResult<UFRGObservation> result = env.reset("easy");
        
        assertNotNull(result.observation);
        assertTrue(result.observation.riskScore() >= 0.0 && result.observation.riskScore() <= 100.0);
        assertEquals("easy", result.info.get("task"));
    }
    
    @Test
    public void testStepCalculatesRewards() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset("easy");
        
        // normal action
        UFRGAction action = new UFRGAction(0, 0, 1); // 0=Approve, 0=Normal, 1=SkipVerify
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(action);
        
        assertTrue(result.reward.value() >= 0.0 && result.reward.value() <= 1.0);
        assertNotNull(result.reward.breakdown());
    }

    @Test
    public void testCircuitBreakerTripped() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset("hard");
        
        UFRGAction action = new UFRGAction(1, 2, 0); // 1=Reject, 2=CircuitBreaker, 0=FullVerify
        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = env.step(action);
        
        assertTrue(result.reward.circuitBreakerTripped());
        assertTrue(result.reward.breakdown().containsKey("circuit_breaker_penalty"));
    }
}
