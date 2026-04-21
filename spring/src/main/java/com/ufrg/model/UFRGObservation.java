package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public record UFRGObservation(
    @JsonProperty("channel") double channel,
    @JsonProperty("risk_score") double riskScore,
    @JsonProperty("kafka_lag") double kafkaLag,
    @JsonProperty("api_latency") double apiLatency,
    @JsonProperty("rolling_p99") double rollingP99
) {}
