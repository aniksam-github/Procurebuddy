package com.procurebuddy.dto.request;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class DocumentSearchRequest {

    @NotBlank
    private String query;
}
