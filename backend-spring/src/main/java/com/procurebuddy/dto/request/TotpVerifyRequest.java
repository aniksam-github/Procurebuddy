package com.procurebuddy.dto.request;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class TotpVerifyRequest {

    @Email
    @NotBlank
    private String email;

    @NotBlank
    private String code;
}
