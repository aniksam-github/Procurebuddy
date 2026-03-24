package com.procurebuddy.controller;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthController {

    @GetMapping("/")
    public Map<String, Object> home() {
        return Map.of("message", "CBRI ProcureBuddy API is running.");
    }

    @GetMapping("/api/health")
    public Map<String, Object> health() {
        return Map.of("ok", true, "timestamp", LocalDateTime.now(ZoneOffset.UTC));
    }
}
