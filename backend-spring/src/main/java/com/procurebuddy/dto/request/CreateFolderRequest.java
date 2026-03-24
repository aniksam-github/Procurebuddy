package com.procurebuddy.dto.request;

import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.Data;

@Data
public class CreateFolderRequest {

    private String user;

    @JsonAlias({"userId", "user_id"})
    private Long userId;

    private String name;
}
