package com.procurebuddy.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record ChatSummaryResponse(
        @JsonProperty("chat_id") String chatId,
        String title,
        String preview,
        @JsonProperty("message_count") long messageCount,
        @JsonProperty("updated_at") LocalDateTime updatedAt,
        @JsonProperty("is_pinned") boolean isPinned,
        @JsonProperty("folder_id") String folderId
) {
}
