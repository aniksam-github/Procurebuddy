package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.Data;

@Data
public class MoveChatRequest {

    private String user;

    @JsonAlias({"userId", "user_id"})
    private Long userId;

    @JsonAlias({"chat_id"})
    private String chatId;

    @JsonAlias({"folder_id"})
    private String folderId;
}
