package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UFRGObservation {
    @JsonProperty("channel")
    private double channel;

    @JsonProperty("risk_score")
    private double riskScore;

    @JsonProperty("kafka_lag")
    private double kafkaLag;

    @JsonProperty("api_latency")
    private double apiLatency;

    @JsonProperty("rolling_p99")
    private double rollingP99;
}
