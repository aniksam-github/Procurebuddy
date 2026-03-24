package com.procurebuddy.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record ChatMessageResponse(String role, String content, LocalDateTime timestamp) {
}
