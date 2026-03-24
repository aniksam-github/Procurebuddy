package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.Data;

@Data
public class PinChatRequest {

    private String user;

    @JsonAlias({"userId", "user_id"})
    private Long userId;

    @JsonAlias({"chat_id"})
    private String chatId;

    @JsonAlias({"is_pinned", "isPinned"})
    private Boolean pinned;
}
