package com.ufrg.env;

import java.util.Map;

public interface GymEnvironment<Obs, Act, Rew> {
    
    class ResetResult<O> {
        public O observation;
        public Map<String, Object> info;
        public ResetResult(O observation, Map<String, Object> info) {
            this.observation = observation;
            this.info = info;
        }
    }

    class StepResult<O, R> {
        public O observation;
        public R reward;
        public boolean done;
        public Map<String, Object> info;
        
        public StepResult(O observation, R reward, boolean done, Map<String, Object> info) {
            this.observation = observation;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }
    }

    ResetResult<Obs> reset(String task);
    StepResult<Obs, Rew> step(Act action);
    Obs state();
}
