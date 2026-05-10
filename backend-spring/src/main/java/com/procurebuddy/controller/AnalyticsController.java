package com.procurebuddy.controller;

import com.procurebuddy.service.PromptAnalyticsService;
import java.security.Principal;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/analytics")
@RequiredArgsConstructor
public class AnalyticsController {

    private final PromptAnalyticsService promptAnalyticsService;

    @GetMapping("/prompts")
    public Map<String, Object> listPromptStats(Principal principal) {
        return promptAnalyticsService.listTopPrompts(principal.getName());
    }
}
