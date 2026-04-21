package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public record UFRGReward(
    @JsonProperty("value") double value,
    @JsonProperty("breakdown") Map<String, Double> breakdown,
    @JsonProperty("crashed") boolean crashed,
    @JsonProperty("circuit_breaker_tripped") boolean circuitBreakerTripped
) {}
