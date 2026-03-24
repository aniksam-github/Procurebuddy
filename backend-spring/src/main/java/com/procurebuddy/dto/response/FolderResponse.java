package com.procurebuddy.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record FolderResponse(
        String id,
        String name,
        @JsonProperty("created_at") LocalDateTime createdAt
) {
}
