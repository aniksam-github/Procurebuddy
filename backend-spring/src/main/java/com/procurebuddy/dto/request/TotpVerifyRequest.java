package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
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

    @JsonAlias("login_token")
    private String loginToken;
}
