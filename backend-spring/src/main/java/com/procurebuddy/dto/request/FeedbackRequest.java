package com.procurebuddy.dto.request;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class FeedbackRequest {

    @NotBlank
    private String user;

    @NotBlank
    private String messageId;

    @NotBlank
    private String type;

    private String chatId;
}
