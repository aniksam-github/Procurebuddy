package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ChangePasswordRequest {

    @Email
    @NotBlank
    private String email;

    @JsonAlias("new_password")
    @NotBlank
    private String newPassword;

    @JsonAlias("login_token")
    private String loginToken;
}
