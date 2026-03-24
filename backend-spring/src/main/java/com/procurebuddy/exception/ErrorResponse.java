package com.procurebuddy.exception;

import java.time.LocalDateTime;

public record ErrorResponse(
        int status,
        String detail,
        LocalDateTime timestamp
) {
}
