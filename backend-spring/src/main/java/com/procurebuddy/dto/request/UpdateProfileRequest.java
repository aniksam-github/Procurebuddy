package com.procurebuddy.dto.request;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class UpdateProfileRequest {

    @Email
    @NotBlank
    private String email;

    @NotBlank
    private String displayName;

    @NotBlank
    private String username;

    private String avatarBase64;
}
