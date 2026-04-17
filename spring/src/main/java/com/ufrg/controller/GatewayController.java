package com.ufrg.controller;

import com.ufrg.env.GymEnvironment;
import com.ufrg.env.UnifiedFintechEnv;
import com.ufrg.model.UFRGAction;
import com.ufrg.model.UFRGObservation;
import com.ufrg.model.UFRGReward;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
public class GatewayController {

    private volatile UnifiedFintechEnv env;

    public GatewayController() {
        this.env = new UnifiedFintechEnv();
    }

    @GetMapping("/")
    public Map<String, String> rootHealthCheck() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("message", "UFRG is live. Use POST /reset to initialise a task.");
        return response;
    }

    @GetMapping("/reset")
    public Map<String, String> resetHealthCheck() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("message", "Route /reset is live. Send POST /reset with {\"task\": \"easy|medium|hard\"} to begin.");
        return response;
    }

    public static class ResetRequest {
        public String task;
    }

    @PostMapping("/reset")
    public Map<String, Object> resetEnv(@RequestBody(required = false) ResetRequest resetRequest) {
        String taskName = "easy";
        if (resetRequest != null && resetRequest.task != null) {
            taskName = resetRequest.task;
        }

        if (!"easy".equals(taskName) && !"medium".equals(taskName) && !"hard".equals(taskName)) {
            throw new IllegalArgumentException("Invalid task '" + taskName + "'. Must be one of: easy, medium, hard.");
        }

        this.env = new UnifiedFintechEnv();
        GymEnvironment.ResetResult<UFRGObservation> result = this.env.reset(taskName);

        Map<String, Object> response = new HashMap<>();
        response.put("observation", result.observation);
        response.put("info", result.info);
        return response;
    }

    public static class StepRequest {
        @Valid
        public UFRGAction action;
    }

    @PostMapping("/step")
    public Map<String, Object> stepEnv(@Valid @RequestBody StepRequest stepRequest) {
        if (stepRequest == null || stepRequest.action == null) {
            throw new IllegalArgumentException("Request body must contain an 'action' key.");
        }

        GymEnvironment.StepResult<UFRGObservation, UFRGReward> result = this.env.step(stepRequest.action);

        Map<String, Object> response = new HashMap<>();
        response.put("observation", result.observation);
        response.put("reward", result.reward.getValue());
        response.put("reward_breakdown", result.reward.getBreakdown());
        response.put("done", result.done);
        response.put("info", result.info);
        return response;
    }

    @GetMapping("/state")
    public Map<String, Object> getState() {
        Map<String, Object> response = new HashMap<>();
        response.put("observation", this.env.state());
        return response;
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> handleExceptions(IllegalArgumentException e) {
        Map<String, String> errorResponse = new HashMap<>();
        errorResponse.put("detail", e.getMessage());
        return ResponseEntity.status(422).body(errorResponse);
    }
}
