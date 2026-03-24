package com.procurebuddy.dto.request;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class TotpEnableRequest {

    @Email
    @NotBlank
    private String email;

    private String secret;

    @NotBlank
    private String code;
}
