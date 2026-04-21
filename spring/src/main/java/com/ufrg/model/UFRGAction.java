package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public record UFRGAction(
    @NotNull @Min(0) @Max(2) @JsonProperty("risk_decision") Integer riskDecision,
    @NotNull @Min(0) @Max(2) @JsonProperty("infra_routing") Integer infraRouting,
    @NotNull @Min(0) @Max(1) @JsonProperty("crypto_verify") Integer cryptoVerify
) {}
