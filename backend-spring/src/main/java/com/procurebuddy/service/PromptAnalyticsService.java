package com.procurebuddy.service;

import com.procurebuddy.entity.PromptStatEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.PromptStatRepository;
import com.procurebuddy.util.UserResolver;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class PromptAnalyticsService {

    private final PromptStatRepository promptStatRepository;
    private final AuthService authService;

    @Transactional
    public void trackPrompt(String promptText) {
        String normalizedPrompt = normalizePrompt(promptText);
        if (normalizedPrompt.isBlank()) {
            return;
        }

        PromptStatEntity stat = promptStatRepository.findByPromptText(normalizedPrompt)
                .orElseGet(PromptStatEntity::new);
        stat.setPromptText(normalizedPrompt);
        stat.setCount(stat.getId() == null ? 1 : stat.getCount() + 1);
        stat.setLastUsedAt(LocalDateTime.now());
        promptStatRepository.save(stat);
    }

    @Transactional(readOnly = true)
    public Map<String, Object> listTopPrompts(String email) {
        String normalizedEmail = UserResolver.normalizeEmail(email);
        if (!authService.isAdminEmail(normalizedEmail)) {
            throw new ApiException(HttpStatus.FORBIDDEN, "Analytics access is restricted to the configured admin account.");
        }

        List<LinkedHashMap<String, Object>> prompts = promptStatRepository.findTop20ByOrderByCountDescLastUsedAtDesc().stream()
                .map(stat -> {
                    LinkedHashMap<String, Object> item = new LinkedHashMap<>();
                    item.put("prompt_text", stat.getPromptText());
                    item.put("count", stat.getCount());
                    item.put("last_used_at", stat.getLastUsedAt());
                    return item;
                })
                .toList();
        return Map.of("success", true, "prompts", prompts, "count", prompts.size());
    }

    private String normalizePrompt(String promptText) {
        if (promptText == null) {
            return "";
        }
        String normalized = promptText.strip().replaceAll("\\s+", " ");
        if (normalized.length() > 1000) {
            normalized = normalized.substring(0, 1000).trim();
        }
        return normalized;
    }
}
