package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class MemoryRequest {

    private String user;

    @JsonAlias({"userId", "user_id"})
    private Long userId;

    @NotBlank
    private String key;

    @NotBlank
    private String value;
}
