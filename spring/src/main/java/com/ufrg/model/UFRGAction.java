package com.ufrg.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UFRGAction {
    @NotNull
    @Min(0)
    @Max(2)
    @JsonProperty("risk_decision")
    private Integer riskDecision;

    @NotNull
    @Min(0)
    @Max(2)
    @JsonProperty("infra_routing")
    private Integer infraRouting;

    @NotNull
    @Min(0)
    @Max(1)
    @JsonProperty("crypto_verify")
    private Integer cryptoVerify;
}
