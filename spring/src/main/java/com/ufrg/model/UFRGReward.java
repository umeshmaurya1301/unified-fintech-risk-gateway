package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UFRGReward {
    @JsonProperty("value")
    private double value;

    @JsonProperty("breakdown")
    private Map<String, Double> breakdown;

    @JsonProperty("crashed")
    private boolean crashed;

    @JsonProperty("circuit_breaker_tripped")
    private boolean circuitBreakerTripped;
}
