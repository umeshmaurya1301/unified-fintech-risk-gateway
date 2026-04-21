package com.ufrg.env;

import com.ufrg.model.UFRGAction;
import com.ufrg.model.UFRGObservation;
import com.ufrg.model.UFRGReward;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@Component
public class UnifiedFintechEnv implements GymEnvironment<UFRGObservation, UFRGAction, UFRGReward> {

    private final int maxSteps = 100;
    
    private double rollingLag = 0.0;
    private double rollingLatency = 50.0;
    
    private int currentStep = 0;
    private String currentTask = "easy";
    private UFRGObservation currentObs;
    private String lastEventType;
    
    private final Random rng = new Random();

    public UnifiedFintechEnv() {
        this.reset("easy");
    }

    @Override
    public ResetResult<UFRGObservation> reset(String task) {
        if (!"easy".equals(task) && !"medium".equals(task) && !"hard".equals(task)) {
            task = "easy";
        }
        this.currentTask = task;
        this.currentStep = 0;
        this.rollingLag = 0.0;
        this.rollingLatency = 50.0;
        
        this.currentObs = generateTransaction(this.currentTask);
        
        Map<String, Object> info = new HashMap<>();
        info.put("task", task);
        
        return new ResetResult<>(this.currentObs, info);
    }

    @Override
    public UFRGObservation state() {
        return this.currentObs;
    }

    private UFRGObservation generateTransaction(String taskName) {
        double channel = (double) rng.nextInt(3); // {0, 1, 2}
        
        double riskScore;
        double kafkaLag;
        double apiLatency;
        String eventType;
        
        if ("easy".equals(taskName)) {
            riskScore = 5.0 + 25.0 * rng.nextDouble();
            kafkaLag = Math.max(0.0, this.rollingLag + (-50.0 + 100.0 * rng.nextDouble()));
            apiLatency = Math.max(10.0, this.rollingLatency + (-30.0 + 60.0 * rng.nextDouble()));
            eventType = "normal";
        } else if ("medium".equals(taskName)) {
            double roll = rng.nextDouble();
            if (roll < 0.80) {
                riskScore = 5.0 + 25.0 * rng.nextDouble();
                kafkaLag = Math.max(0.0, this.rollingLag + (-50.0 + 100.0 * rng.nextDouble()));
                apiLatency = Math.max(10.0, this.rollingLatency + (-30.0 + 60.0 * rng.nextDouble()));
                eventType = "normal";
            } else {
                riskScore = 10.0 * rng.nextDouble();
                this.rollingLag += 500.0 + 500.0 * rng.nextDouble();
                this.rollingLatency += 100.0 + 200.0 * rng.nextDouble();
                kafkaLag = this.rollingLag + 200.0 * rng.nextDouble();
                apiLatency = this.rollingLatency + 100.0 * rng.nextDouble();
                eventType = "flash_sale";
            }
        } else if ("hard".equals(taskName)) {
            riskScore = 85.0 + 15.0 * rng.nextDouble();
            this.rollingLag += 100.0 + 300.0 * rng.nextDouble();
            this.rollingLatency += 50.0 + 100.0 * rng.nextDouble();
            kafkaLag = this.rollingLag + 300.0 * rng.nextDouble();
            apiLatency = this.rollingLatency + 200.0 * rng.nextDouble();
            eventType = "botnet_attack";
        } else {
            throw new IllegalArgumentException("Unknown task: " + taskName);
        }
        
        double alpha = 0.2;
        this.rollingLag = alpha * kafkaLag + (1.0 - alpha) * this.rollingLag;
        this.rollingLatency = alpha * apiLatency + (1.0 - alpha) * this.rollingLatency;
        
        double smoothedP99 = Math.min(this.rollingLatency, 5000.0);
        
        kafkaLag = clip(kafkaLag, 0.0, 10000.0);
        apiLatency = clip(apiLatency, 0.0, 5000.0);
        riskScore = clip(riskScore, 0.0, 100.0);
        channel = clip(channel, 0.0, 2.0);
        
        this.lastEventType = eventType;
        
        return new UFRGObservation(channel, riskScore, kafkaLag, apiLatency, smoothedP99);
    }
    
    private double clip(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }

    @Override
    public StepResult<UFRGObservation, UFRGReward> step(UFRGAction action) {
        double riskScore = this.currentObs.riskScore();
        double kafkaLag = this.currentObs.kafkaLag();
        double rollingP99 = this.currentObs.rollingP99();
        String currentEventType = this.lastEventType;

        boolean circuitBreakerTripped = false;
        boolean done = false;

        if (action.cryptoVerify() == 0) {
            this.rollingLag += 150.0;
            this.rollingLatency += 200.0;
        } else {
            this.rollingLag -= 100.0;
        }

        if (action.infraRouting() == 0) {
            this.rollingLag += 100.0;
        } else if (action.infraRouting() == 1) {
            this.rollingLag -= 300.0;
        } else if (action.infraRouting() == 2) {
            this.rollingLag = 0.0;
            this.rollingLatency = 50.0;
            circuitBreakerTripped = true;
        }

        this.rollingLag = Math.max(0.0, this.rollingLag);
        this.rollingLatency = Math.max(0.0, this.rollingLatency);

        double reward = 0.8;

        if (action.infraRouting() == 1) {
            if ("flash_sale".equals(currentEventType)) {
                reward -= 0.1;
            } else {
                reward -= 0.2;
            }
        }

        if (rollingP99 > 800.0) {
            reward -= 0.3;
        } else if (rollingP99 > 500.0 && rollingP99 <= 800.0) {
            double proximity = (rollingP99 - 500.0) / 300.0;
            reward -= 0.1 * proximity;
        }

        if (circuitBreakerTripped) {
            reward -= 0.5;
        }

        if (this.rollingLag > 3000.0 && this.rollingLag <= 4000.0 && !circuitBreakerTripped) {
            double proximity = (this.rollingLag - 3000.0) / 1000.0;
            reward -= 0.1 * proximity;
        }

        if (riskScore > 80.0 && action.riskDecision() == 2) {
            reward += 0.05;
        }

        if (riskScore > 80.0 && action.cryptoVerify() == 0) {
            reward += 0.03;
        }

        if (action.cryptoVerify() == 1 && action.riskDecision() == 0 && riskScore > 80.0) {
            reward -= 1.0;
        }
        
        if (this.rollingLag > 4000.0 && !circuitBreakerTripped) {
            reward = 0.0;
            done = true;
        }
        
        this.currentStep += 1;
        this.currentObs = generateTransaction(this.currentTask);
        
        if (this.currentStep >= this.maxSteps && !done) {
            done = true;
        }
        
        double finalReward = Math.max(0.0, Math.min(1.0, reward));
        
        Map<String, Double> breakdown = new HashMap<>();
        breakdown.put("baseline", 0.8);
        if (action.infraRouting() == 1) {
            if ("flash_sale".equals(currentEventType)) breakdown.put("throttle_flash_sale_penalty", -0.1);
            else breakdown.put("throttle_penalty", -0.2);
        }
        if (rollingP99 > 800.0) breakdown.put("sla_breach_penalty", -0.3);
        else if (rollingP99 > 500.0 && rollingP99 <= 800.0) breakdown.put("sla_proximity_warning", -0.1 * ((rollingP99 - 500.0) / 300.0));

        if (circuitBreakerTripped) breakdown.put("circuit_breaker_penalty", -0.5);

        if (this.rollingLag > 3000.0 && this.rollingLag <= 4000.0 && !circuitBreakerTripped) {
            breakdown.put("lag_proximity_warning", -0.1 * ((this.rollingLag - 3000.0) / 1000.0));
        }

        if (riskScore > 80.0 && action.riskDecision() == 2) breakdown.put("challenge_bonus", 0.05);
        if (riskScore > 80.0 && action.cryptoVerify() == 0) breakdown.put("fullverify_bonus", 0.03);

        if (action.cryptoVerify() == 1 && action.riskDecision() == 0 && riskScore > 80.0) breakdown.put("fraud_penalty", -1.0);
        if (this.rollingLag > 4000.0 && !circuitBreakerTripped) breakdown.put("crash_override", 0.0);
        
        boolean crashed = this.rollingLag > 4000.0 && !circuitBreakerTripped;
        UFRGReward typedReward = new UFRGReward(finalReward, breakdown, crashed, circuitBreakerTripped);
        
        Map<String, Object> info = new HashMap<>();
        info.put("step", this.currentStep);
        info.put("task", this.currentTask);
        info.put("event_type", currentEventType);
        info.put("obs_risk_score", riskScore);
        info.put("obs_kafka_lag", kafkaLag);
        info.put("obs_rolling_p99", rollingP99);
        info.put("action_risk_decision", action.riskDecision());
        info.put("action_infra_routing", action.infraRouting());
        info.put("action_crypto_verify", action.cryptoVerify());
        info.put("reward_raw", reward);
        info.put("reward_final", finalReward);
        info.put("circuit_breaker_tripped", circuitBreakerTripped);
        info.put("crashed", crashed);
        info.put("done", done);
        info.put("internal_rolling_lag", this.rollingLag);
        info.put("internal_rolling_latency", this.rollingLatency);
        
        return new StepResult<>(this.currentObs, typedReward, done, info);
    }
}
